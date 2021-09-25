from itertools import islice

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from model_repair.cache42.cached_42 import cache_42
import tools.pudb_hook

from tools.utils import write_json, write_json_np_ok, read_json
from model_repair.override_for_release import get_interface


# @cache_42(ignore_args=["gcfg", "verbose"], force_recompute=True)
@cache_42(ignore_args=["gcfg", "verbose"])
def finetune_on_cluster_ewc(cfg, split_out, cluster_id, verbose, gcfg, cache):

    if verbose:
        print(f"\n\n\n ******************************************** Starting finetuning on {cluster_id} ******************************************** \n")

    # Load dataframe
    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info_old = pd.read_feather(str(path_df))

    # Load interface to get split algorithm adapted to the task
    interface = get_interface(gcfg)

    # Get the model
    if gcfg["trigger_extract"].epochs is None or len(gcfg["trigger_extract"].epochs) > 1:
        # If 1 epoch, it can be the ref epoch..
        epoch = int(gcfg["trigger_extract"]["ref_epoch"])
    elif len(gcfg["trigger_extract"].epochs) == 1:
        epoch = gcfg["trigger_extract"].epochs[0]

    checkpoint_path = interface.get_checkpoint_path(epoch)
    model = interface.load_model(checkpoint_path)

    # Create output path
    out_path = cache.get_path()
    out_path.mkdir(parents=True, exist_ok=True)
    filename_best_val_model = str(out_path / "model_best_val.pytorch")

    assert cluster_id is not None # NOTE: I used to use None to mean all clusters but that's just confusing now

    # EWC set-up --------------------
    fisher_dict = {}
    optpar_dict = {}

    if "alternative_ewc" in cfg:
        # pass
        if cfg["alternative_ewc"] == "continual_ai":
            print("EWC continual_ai")
            get_fisher_and_opt_params_continual_ai(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id="correct")
        elif cfg["alternative_ewc"] == "moskomule":
            print("EWC moskomule")
            get_fisher_and_opt_params_github_moskomule(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id="correct")
        else:
            raise NotImplementedError()
    else:
        get_fisher_and_opt_params(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id="correct")
    
    previous_tasks = ["correct"]



    if cluster_id == "all":

        # Select all objects which are non -1 (non-mistakes)
        objs_info_old_f = objs_info_old[objs_info_old["cluster"]!=-1]

        train_info = objs_info_old_f[(objs_info_old_f["split_id"]==0)]
        val_info = objs_info_old_f[(objs_info_old_f["split_id"]==1)]

        n_clusters = objs_info_old["cluster"].unique().max() + 1
        max_iter_mult_factor = n_clusters # If 5 clusters, training on all should do 5x the iters of training on one cluster

    elif cluster_id == "nothing":
        # Save current model and return (no finetuning)
        torch.save(model.state_dict(), filename_best_val_model)
        return cache

    elif cluster_id == "other_epoch":
        # Get the model from another epoch
        checkpoint_path = interface.get_checkpoint_path(cfg["other_epoch"])
        model = interface.load_model(checkpoint_path)
        torch.save(model.state_dict(), filename_best_val_model)
        return cache

    elif isinstance(cluster_id, int):
        objs_info_old_f = objs_info_old[objs_info_old["cluster"]==cluster_id]

        train_info = objs_info_old_f[(objs_info_old_f["split_id"]==0)]
        val_info = objs_info_old_f[(objs_info_old_f["split_id"]==1)]

        max_iter_mult_factor = 1.0 # If 5 clusters, training on all should do 5x the iters of training on one cluster
    else:
        raise NotImplementedError()

    # Get the dataset
    example_key = interface.get_example_key()

    specific_frames_cluster_train = train_info[example_key].unique().tolist()
    specific_frames_cluster_val = val_info[example_key].unique().tolist()


    if "prevent_tp_fpfn_overlap" in cfg and cfg["prevent_tp_fpfn_overlap"]:
        print("prevent_tp_fpfn_overlap mode !")
        specific_frames_correct_train, specific_frames_correct_val, specific_frames_correct_test = get_list_of_correct_frames(objs_info_old, example_key)

        all_correct_frames = specific_frames_correct_train + specific_frames_correct_val + specific_frames_correct_test

        specific_frames_cluster_train = list(set(specific_frames_cluster_train) - set(all_correct_frames))
        specific_frames_cluster_val = list(set(specific_frames_cluster_val) - set(all_correct_frames))

    dataloader_train = interface.get_dataloader(specific_frames=specific_frames_cluster_train)
    dataloader_val = interface.get_dataloader(specific_frames=specific_frames_cluster_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_val = -1
    epoch_last_inc_val = 0

    # NOTE: Start by saving not finetuned model so correct cluster early stopping can stop event at first epoch
    torch.save(model.state_dict(), filename_best_val_model)

    if "skip_finetuning" in cfg and cfg["skip_finetuning"]:
        return cache

    if "how_many_steps_to" in cfg:
        # cfg["how_many_steps_to"]
        how_many_steps_to = {k: np.inf for k in cfg["how_many_steps_to"]}
        filename_how_many_steps_to = str(out_path / "how_many_steps_to.json")

    stats_finetuning = {"train": {}, "val": {}}
    filename_stats_finetuning = str(out_path / "stats_finetuning.json")

    if "override_val_metric" in cfg:
        print("override_val_metric")
        val_metric = cfg["override_val_metric"]
    else:
        val_metric = interface.validation_metric()

    if "train_grad_pos" not in cfg:
        train_grad_pos = True
    else:
        train_grad_pos = cfg["train_grad_pos"]
        if not train_grad_pos:
            print("WARNING: no train_grad_pos_mode")
            
    for epoch in range(cfg["max_epochs_train"]):
        if verbose:
            print(f"------------------------------------------ Epoch: {epoch} ------------------------------------------------------")

        # Train : -----------------------------------------------------------------------------------------------------------------

        torch.set_grad_enabled(True)
        model.train()

        keep_penalties = []
        all_objs = []
        sliced_dataloader_train = islice(dataloader_train, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))
        for i_batch, sample in tqdm(enumerate(sliced_dataloader_train), total=len(dataloader_train)):

            optimizer.zero_grad()

            # Get the list of objects on this frame that belong to the same cluster and train split
            example_key = interface.get_example_key()
            this_cluster_objs = train_info[train_info[example_key]==sample[example_key][0] ]
  
            first_loss_save = None
            for idx, train_this in this_cluster_objs.iterrows():

                if interface.requires_grad_pos() and train_grad_pos:
                    grad_pos = (train_this["r"], train_this["t"])
                else:
                    grad_pos = None

                loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)

                # EWC penalty --------------------------------------------------------------
                penalty = 0.0
                for previous_cluster_id in previous_tasks:
                    for name, param in model.named_parameters():
                        fisher = fisher_dict[previous_cluster_id][name]
                        optpar = optpar_dict[previous_cluster_id][name]
                        
                        if "ewc_prior" in cfg:
                            penal_fish = (fisher * (optpar - param).pow(2)).sum()
                            penal_prior = (cfg["ewc_prior"] * (optpar - param).pow(2)).sum()
                            penalty += (penal_fish + penal_prior) * cfg["ewc_lambda"] * 0.5
                        else:
                            penalty += (fisher * (optpar - param).pow(2)).sum() * cfg["ewc_lambda"]

                keep_penalties.append(penalty.item())
                loss = penalty + loss

                loss.backward()

                if first_loss_save is None:
                    first_loss_save = loss.cpu().detach().numpy()

                if not train_grad_pos: # Only one iter
                    assert grad_pos is None
                    break

            optimizer.step()
            
            # Get the (new) objects for computing the metrics
            objs = interface.get_objects(preds, sample) # List of dicts
            objs = pd.DataFrame(objs)

            # Identify the cluster for each object (-2 means new object, -1 correct cluster)
            objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

            # Filter objects to only keep objects belonging to the current cluster
            if cluster_id == "all":
                objs = objs[objs["cluster"] >= 0]
            else:
                objs = objs[objs["cluster"] == cluster_id]
            objs["loss"] = first_loss_save
            all_objs.append(objs)

        all_objs = pd.concat(all_objs, 0).reset_index(drop=True)
        m = interface.compute_metrics(pd.DataFrame(all_objs))
        m["penalty_ewc"] = np.mean(keep_penalties)
        if verbose:
            print(f"Train, {m}")

        stats_finetuning["train"][epoch] = {
            "all": None,
            "correct": None,
            "cluster": m,
        }

        stats_finetuning["val"][epoch] = {
            "all": None,
            "correct": None,
            "cluster": None,
        }

        still_ok, correct_cluster_perf, m_correct = check_correct_cluster_perf(cfg, objs_info_old, interface, model)
        stats_finetuning["val"][epoch]["correct"] = m_correct

        preserve_correct_ok = still_ok

        # Val : -----------------------------------------------------------------------------------------------------------------
        torch.set_grad_enabled(False)
        model.eval()

        all_objs = []
        sliced_dataloader_val = islice(dataloader_val, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))
        for i_batch, sample in tqdm(enumerate(sliced_dataloader_val), total=len(dataloader_val)):

            loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)
            loss = loss.cpu().detach().numpy() # Note that value of loss for the val set is different for object detection as I use grad_pos=None

            # Get the (new) objects for computing the metrics
            objs = interface.get_objects(preds, sample) # List of dicts
            objs = pd.DataFrame(objs)

            # Identify the cluster for each object (-2 means new object, -1 correct cluster)
            objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

            # Filter objects to only keep objects belonging to the current cluster
            if cluster_id == "all":
                objs = objs[objs["cluster"] >= 0]
            else:
                objs = objs[objs["cluster"] == cluster_id]

            objs["loss"] = loss

            all_objs.append(objs)

        all_objs = pd.concat(all_objs, 0).reset_index(drop=True)
        m = interface.compute_metrics(pd.DataFrame(all_objs))
        if verbose:
            print(f"Val, {m}")
        validation_value = m[val_metric]

        stats_finetuning["val"][epoch]["cluster"] = m
        
        if (validation_value > best_val or best_val < 0) and preserve_correct_ok:
            best_val = validation_value
            epoch_last_inc_val = epoch
            if verbose:
                print(f"New best val !!!! -> {validation_value}")
                print(f"Saving model to {filename_best_val_model}")
            torch.save(model.state_dict(), filename_best_val_model)

            if "how_many_steps_to" in cfg:
                for thresh in how_many_steps_to.keys():
                    if validation_value > thresh:
                        how_many_steps_to[thresh] = epoch * max_iter_mult_factor * cfg["max_steps_per_epoch"]
                        write_json(filename_how_many_steps_to, how_many_steps_to)

        write_json_np_ok(filename_stats_finetuning, stats_finetuning)
        if epoch - epoch_last_inc_val > cfg["max_wait_epoch_inc"]:
            break


# EWC specific -----------------------------------------------------------------------------------------------
def get_info_df(objs_info_old, cluster_id):
    if cluster_id == "all":

        # Select all objects which are non -1 (non-mistakes)
        objs_info_old_f = objs_info_old[objs_info_old["cluster"]!=-1]

        train_info = objs_info_old_f[(objs_info_old_f["split_id"]==0)]
        val_info = objs_info_old_f[(objs_info_old_f["split_id"]==1)]
        test_info = None

    elif cluster_id == "correct":

        # Select all objects which are -1 or -2
        objs_info_old_f = objs_info_old[objs_info_old["cluster"]<0]

        train_info = objs_info_old_f[(objs_info_old_f["split_id"]==0)]
        val_info = objs_info_old_f[(objs_info_old_f["split_id"]==1)]

        test_info = objs_info_old_f[(objs_info_old_f["split_id"]==2)]

    elif isinstance(cluster_id, int):
        objs_info_old_f = objs_info_old[objs_info_old["cluster"]==cluster_id]

        train_info = objs_info_old_f[(objs_info_old_f["split_id"]==0)]
        val_info = objs_info_old_f[(objs_info_old_f["split_id"]==1)]
        test_info = None
    else:
        raise NotImplementedError()

    return train_info, val_info, test_info


def get_fisher_and_opt_params(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id):
    
    train_info, val_info, test_info = get_info_df(objs_info_old, cluster_id)

    if "use_tns_list" in cfg:
        raise NotImplementedError()

    # Get the dataset
    example_key = interface.get_example_key()
    dataloader_train = interface.get_dataloader(specific_frames=train_info[example_key].unique().tolist())

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    torch.set_grad_enabled(True)
    model.train()

    all_objs = []
    sliced_dataloader_train = islice(dataloader_train, cfg["n_samples"])
    count = 0
    for i_batch, sample in tqdm(enumerate(sliced_dataloader_train), total=len(dataloader_train)):

        optimizer.zero_grad()

        # Get the list of objects on this frame that belong to the same cluster and train split
        example_key = interface.get_example_key()
        this_cluster_objs = train_info[train_info[example_key]==sample[example_key][0] ]

        for idx, train_this in this_cluster_objs.iterrows():

            if interface.requires_grad_pos():
                grad_pos = (train_this["r"], train_this["t"])
            else:
                grad_pos = None

            loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)
            loss.backward()

        count += 1
        if cluster_id not in optpar_dict:
            optpar_dict[cluster_id] = {}
            fisher_dict[cluster_id] = {}

        for name, param in model.named_parameters():
            if name not in optpar_dict[cluster_id]:
                optpar_dict[cluster_id][name] = param.data.clone() # Only once (all the same)
                fisher_dict[cluster_id][name] = param.grad.data.clone().pow(2)
            else:
                fisher_dict[cluster_id][name] += param.grad.data.clone().pow(2)

    # Divide the sum for average (expected value)
    for name, param in model.named_parameters():
        fisher_dict[cluster_id][name] /= count
        print(f"Fishier {name}", fisher_dict[cluster_id][name].mean())


# ----------------------------------------------------------------------------------------

def check_correct_cluster_perf(cfg, objs_info_old, interface, model):

    if not "check_corrrect_use" in cfg or cfg["check_corrrect_use"] is False:
        return True, None, None

    if "override_val_metric" in cfg:
        val_metric = cfg["override_val_metric"]
    else:
        val_metric = interface.validation_metric()

    cluster_id = "correct"

    train_info, val_info, test_info = get_info_df(objs_info_old, cluster_id)

    # Get the dataset
    example_key = interface.get_example_key()

    specific_frames_correct_val = val_info[example_key].unique().tolist()
    if "use_tns_list" in cfg:
        specific_frames_correct_val += read_json(cfg["use_tns_list"])["val"]

    dataloader_val = interface.get_dataloader(specific_frames=specific_frames_correct_val)


    torch.set_grad_enabled(False)
    model.eval()

    all_objs = []
    sliced_dataloader_val = islice(dataloader_val, cfg["check_correct_n_samples"])

    for i_batch, sample in tqdm(enumerate(sliced_dataloader_val), total=len(dataloader_val)):


        loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)
        loss = loss.cpu().detach().numpy() # Note that value of loss for the val set is different for object detection as I use grad_pos=None

        # Get the (new) objects for computing the metrics
        objs = interface.get_objects(preds, sample) # List of dicts
        objs = pd.DataFrame(objs)

        # Identify the cluster for each object (-2 means new object, -1 correct cluster)
        objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

        # Filter objects to only keep objects belonging to the current cluster

        if len(objs) > 0:
            objs = objs[objs["cluster"] == -1] # NOTE: Only correct cluster objects in this function !
        objs["loss"] = loss
        all_objs.append(objs)


    all_objs = pd.concat(all_objs, 0).reset_index(drop=True)
    m = interface.compute_metrics(pd.DataFrame(all_objs))
    correct_cluster_perf = m[val_metric]

    if correct_cluster_perf < cfg["check_corrrect_thresh"]:
        return False, correct_cluster_perf, m
    else:
        return True, correct_cluster_perf, m


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Try to do like them: https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb

def get_fisher_and_opt_params_continual_ai(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id):
    pass
    # TODO do pass on cluster task_id and update dicts
    
    print(f"get_fisher_and_opt_params for cluster {cluster_id} !!!")
    train_info, val_info, test_info = get_info_df(objs_info_old, cluster_id)

    if "use_tns_list" in cfg:
        raise NotImplementedError()

    # Get the dataset
    example_key = interface.get_example_key()
    dataloader_train = interface.get_dataloader(specific_frames=train_info[example_key].unique().tolist())

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    torch.set_grad_enabled(True)
    model.train()

    all_objs = []
    sliced_dataloader_train = islice(dataloader_train, cfg["n_samples"])

    for i_batch, sample in tqdm(enumerate(sliced_dataloader_train), total=len(dataloader_train)):

        # Get the list of objects on this frame that belong to the same cluster and train split
        example_key = interface.get_example_key()
        this_cluster_objs = train_info[train_info[example_key]==sample[example_key][0] ]

        for idx, train_this in this_cluster_objs.iterrows():

            if interface.requires_grad_pos():
                grad_pos = (train_this["r"], train_this["t"])
            else:
                grad_pos = None

            loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)
            loss.backward()

    if cluster_id not in optpar_dict:
        optpar_dict[cluster_id] = {}
        fisher_dict[cluster_id] = {}

    for name, param in model.named_parameters():
        if name not in optpar_dict[cluster_id]:
            optpar_dict[cluster_id][name] = param.data.clone() # Only once (all the same)
            fisher_dict[cluster_id][name] = param.grad.data.clone().pow(2)

# *****************************************************************************************************************************


def get_fisher_and_opt_params_github_moskomule(cfg, fisher_dict, optpar_dict, objs_info_old, interface, model, cluster_id):
    
    train_info, val_info, test_info = get_info_df(objs_info_old, cluster_id)

    # Get the dataset
    example_key = interface.get_example_key()

    specific_frames_correct_train = train_info[example_key].unique().tolist()
    if "use_tns_list" in cfg:
        specific_frames_correct_train += read_json(cfg["use_tns_list"])["train"]

    dataloader_train = interface.get_dataloader(specific_frames=specific_frames_correct_train)

    if "train_grad_pos" not in cfg:
        train_grad_pos = True
    else:
        train_grad_pos = cfg["train_grad_pos"]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    torch.set_grad_enabled(True)
    model.train()

    all_objs = []
    sliced_dataloader_train = islice(dataloader_train, cfg["n_samples"])
    count = 0

    n_samples = cfg["n_samples"]

    for i_batch, sample in tqdm(enumerate(sliced_dataloader_train), total=len(dataloader_train)):

        optimizer.zero_grad()

        # Get the list of objects on this frame that belong to the same cluster and train split
        example_key = interface.get_example_key()
        this_cluster_objs = train_info[train_info[example_key]==sample[example_key][0] ]

        for idx, train_this in this_cluster_objs.iterrows():

            if interface.requires_grad_pos() and train_grad_pos:
                grad_pos = (train_this["r"], train_this["t"])
            else:
                grad_pos = None

            loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)
            loss.backward()

        if len(this_cluster_objs) == 0 and "use_tns_list" in cfg: # Fix for original TNs
            assert not train_grad_pos
            grad_pos = None

            loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)
            loss.backward()


        count += 1
        if cluster_id not in optpar_dict:
            optpar_dict[cluster_id] = {}
            fisher_dict[cluster_id] = {}

        for name, param in model.named_parameters():
            if name not in optpar_dict[cluster_id]:
                optpar_dict[cluster_id][name] = param.detach().data.clone() # Only once (all the same)
                fisher_dict[cluster_id][name] = param.grad.data.detach().clone().pow(2) / n_samples
            else:
                fisher_dict[cluster_id][name] += param.grad.data.detach().clone().pow(2) / n_samples

def get_list_of_correct_frames(objs_info_old, example_key):

    train_info, val_info, test_info = get_info_df(objs_info_old, "correct")

    specific_frames_correct_train = train_info[example_key].unique().tolist()
    specific_frames_correct_val = val_info[example_key].unique().tolist()
    specific_frames_correct_test = test_info[example_key].unique().tolist()
    
    return specific_frames_correct_train, specific_frames_correct_val, specific_frames_correct_test