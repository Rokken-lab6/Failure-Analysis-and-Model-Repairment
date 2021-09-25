
from itertools import islice

from tqdm import tqdm
import pandas as pd
import torch
import itertools

from model_repair.override_for_release import get_interface
from tools.utils import write_json, write_json_np_ok, read_json

from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg", "verbose"], force_recompute=True)
@cache_42(ignore_args=["gcfg", "verbose"])
def finetune_on_cluster(cfg, split_out, cluster_id, verbose, gcfg, cache):

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

    example_key = interface.get_example_key()

    if "override_max_iter_mult_factor" in cfg:
        max_iter_mult_factor = cfg["override_max_iter_mult_factor"]
        
    if "override_val_metric" in cfg:
        print("override_val_metric")
        val_metric = cfg["override_val_metric"]
    else:
        val_metric = interface.validation_metric()

    if "train_also_correct" in cfg and cfg["train_also_correct"]:
        train_also_correct = True
    else:
        train_also_correct = False

    if "val_also_correct" in cfg and cfg["val_also_correct"]:
        val_also_correct = True
    else:
        val_also_correct = False

    if "train_also_correct_global_val" in cfg and cfg["train_also_correct_global_val"]:
        print("train_also_correct_global_val ! ")
        train_also_correct_global_val = True
    else:
        train_also_correct_global_val = False

    if train_also_correct or val_also_correct:
        objs_info_old_f_correct_cluster = objs_info_old[objs_info_old["cluster"]<0]

        train_info_correct = objs_info_old_f_correct_cluster[(objs_info_old_f_correct_cluster["split_id"]==0)]
        val_info_correct = objs_info_old_f_correct_cluster[(objs_info_old_f_correct_cluster["split_id"]==1)]
        test_info_correct = objs_info_old_f_correct_cluster[(objs_info_old_f_correct_cluster["split_id"]==2)]


        specific_frames_correct_train = train_info_correct[example_key].unique().tolist()
        specific_frames_correct_val = val_info_correct[example_key].unique().tolist()
        specific_frames_correct_test = test_info_correct[example_key].unique().tolist()

        if "use_tns_list" in cfg:
            specific_frames_correct_train += read_json(cfg["use_tns_list"])["train"]
            specific_frames_correct_val += read_json(cfg["use_tns_list"])["val"]
            specific_frames_correct_test += read_json(cfg["use_tns_list"])["test"]

        all_correct_frames = specific_frames_correct_train + specific_frames_correct_val + specific_frames_correct_test

        dataloader_train_correct = interface.get_dataloader(specific_frames=specific_frames_correct_train)
        dataloader_val_correct = interface.get_dataloader(specific_frames=specific_frames_correct_val)


    specific_frames_cluster_train = train_info[example_key].unique().tolist()
    specific_frames_cluster_val = val_info[example_key].unique().tolist()
    if "prevent_tp_fpfn_overlap" in cfg and cfg["prevent_tp_fpfn_overlap"]:
        
        # Remove all the frames in correct frames from cluster frames (in case of double counting due to object level)
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
        torch.save(model.state_dict(), filename_best_val_model)
        return cache

    if "how_many_steps_to" in cfg:
        how_many_steps_to = {k: -1 for k in cfg["how_many_steps_to"]}
        filename_how_many_steps_to = str(out_path / "how_many_steps_to.json")

    stats_finetuning = {"train": {}, "val": {}}
    filename_stats_finetuning = str(out_path / "stats_finetuning.json")

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
        all_objs = []
        sliced_dataloader_train = islice(dataloader_train, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))

        if train_also_correct:
    
            sliced_dataloader_train_correct = islice(dataloader_train_correct, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))

            final_sliced_dataloader = itertools.chain.from_iterable(zip(sliced_dataloader_train, sliced_dataloader_train_correct))
            total_len = len(dataloader_train) + len(dataloader_train_correct)

        else:
            final_sliced_dataloader = sliced_dataloader_train
            total_len = len(dataloader_train)
            
        is_cluster_train = True
        for i_batch, sample in tqdm(enumerate(final_sliced_dataloader), total=total_len):

            optimizer.zero_grad()

            # Get the list of objects on this frame that belong to the same cluster and train split
            example_key = interface.get_example_key()

            if is_cluster_train:
                this_cluster_objs = train_info[train_info[example_key]==sample[example_key][0] ]
            else:
                this_cluster_objs = train_info_correct[train_info_correct[example_key]==sample[example_key][0] ]
            
            if "use_tns_list" not in cfg:
                assert len(this_cluster_objs) > 0

            first_loss_save = None
            for idx, train_this in this_cluster_objs.iterrows():

                if interface.requires_grad_pos() and train_grad_pos:
                    grad_pos = (train_this["r"], train_this["t"])
                else:
                    grad_pos = None

                loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)

                if is_cluster_train:
                    loss = cfg.train_also_correct_weight_cluster * loss
                loss.backward()

                if first_loss_save is None:
                    first_loss_save = loss.cpu().detach().numpy()
                
                if not train_grad_pos: # Only one iter
                    assert grad_pos is None
                    break

            if len(this_cluster_objs) == 0 and "use_tns_list" in cfg: # Fix for original TNs

                assert not train_grad_pos
                grad_pos = None

                loss, preds = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=True)

                if is_cluster_train:
                    loss = cfg.train_also_correct_weight_cluster * loss
                loss.backward()

                if first_loss_save is None:
                    first_loss_save = loss.cpu().detach().numpy()
           

            optimizer.step()

            # Get the (new) objects for computing the metrics
            objs = interface.get_objects(preds, sample) # List of dicts
            objs = pd.DataFrame(objs)

            # Identify the cluster for each object (-2 means new object, -1 correct cluster)
            objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

            # Filter objects to only keep objects belonging to the current cluster
            if train_also_correct:
                if cluster_id == "all":
                    pass
                else:
                    objs = objs[(objs["cluster"] == cluster_id) | (objs["cluster"] < 0)]
            else:
                if cluster_id == "all":
                    objs = objs[objs["cluster"] >= 0]
                else:
                    objs = objs[objs["cluster"] == cluster_id]
            objs["loss"] = first_loss_save
            all_objs.append(objs)

            if train_also_correct:
                is_cluster_train = not is_cluster_train

        all_objs = pd.concat(all_objs, 0).reset_index(drop=True)
        all_objs = pd.DataFrame(all_objs)
        m = interface.compute_metrics(all_objs)
        if verbose:
            print(f"Train, {m}")

        stats_finetuning["train"][epoch] = {
            "all": m,
            "correct": None,
            "cluster": None,
        }

        if train_also_correct:
            correct_cluster_objs = all_objs[all_objs["cluster"] < 0]

            if cluster_id == "all":
                current_cluster_objs = all_objs[all_objs["cluster"] >= 0]
            else:
                current_cluster_objs = all_objs[all_objs["cluster"] == cluster_id]

            correct_c_m = interface.compute_metrics(correct_cluster_objs)
            current_c_m = interface.compute_metrics(current_cluster_objs)

            stats_finetuning["train"][epoch]["correct"] = correct_c_m
            stats_finetuning["train"][epoch]["cluster"] = current_c_m

            if verbose:
                print(f"Train current cluster, {current_c_m}")
                print(f"Train correct, {correct_c_m}")

        # Val : -----------------------------------------------------------------------------------------------------------------
        torch.set_grad_enabled(False)
        model.eval()
        all_objs = []
        sliced_dataloader_val = islice(dataloader_val, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))

        if train_also_correct or val_also_correct:
            sliced_dataloader_val_correct = islice(dataloader_val_correct, int(max_iter_mult_factor*cfg["max_steps_per_epoch"]))

            final_sliced_dataloader = itertools.chain.from_iterable(zip(sliced_dataloader_val, sliced_dataloader_val_correct))
            total_len = len(dataloader_val) + len(dataloader_val_correct)
        else:
            final_sliced_dataloader = sliced_dataloader_val
            total_len = len(dataloader_val)

        for i_batch, sample in tqdm(enumerate(final_sliced_dataloader), total=total_len):

            loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)
            loss = loss.cpu().detach().numpy() # Note that value of loss for the val set is different for object detection as I use grad_pos=None

            # Get the (new) objects for computing the metrics
            objs = interface.get_objects(preds, sample) # List of dicts
            objs = pd.DataFrame(objs)

            # Identify the cluster for each object (-2 means new object, -1 correct cluster)
            objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

            # Filter objects to only keep objects belonging to the current cluster
            if train_also_correct or val_also_correct:
                if cluster_id == "all":
                    pass
                else:
                    if len(objs) > 0:
                        objs = objs[(objs["cluster"] == cluster_id) | (objs["cluster"] < 0)]
            else:
                if cluster_id == "all":
                    objs = objs[objs["cluster"] >= 0]
                else:
                    objs = objs[objs["cluster"] == cluster_id]
            objs["loss"] = loss

            all_objs.append(objs)

        all_objs = pd.concat(all_objs, 0).reset_index(drop=True)
        all_objs = pd.DataFrame(all_objs)
        m = interface.compute_metrics(all_objs)
        if verbose:
            print(f"Val, {m}")
        
        stats_finetuning["val"][epoch] = {
            "all": m,
            "correct": None,
            "cluster": None,
        }

        if train_also_correct or val_also_correct:
            correct_cluster_objs = all_objs[all_objs["cluster"] < 0]

            if cluster_id == "all":
                current_cluster_objs = all_objs[all_objs["cluster"] >= 0]
            else:
                current_cluster_objs = all_objs[all_objs["cluster"] == cluster_id]

            correct_c_m = interface.compute_metrics(correct_cluster_objs)
            current_c_m = interface.compute_metrics(current_cluster_objs)

            stats_finetuning["val"][epoch]["correct"] = correct_c_m
            stats_finetuning["val"][epoch]["cluster"] = current_c_m

            if verbose:
                print(f"Val current cluster, {current_c_m}")
                print(f"Val correct, {correct_c_m}")

            if train_also_correct_global_val:
                validation_value = m[val_metric]
                preserve_correct_ok = True
            else:
                correct_c_validation_value = correct_c_m[val_metric]
                validation_value = current_c_m[val_metric]

                preserve_correct_ok = (correct_c_validation_value >= cfg["train_also_correct_preserve_thresh"])

        else:
            validation_value = m[val_metric]
            preserve_correct_ok = True

        
        if (validation_value > best_val or best_val < 0) and preserve_correct_ok:
            best_val = validation_value
            epoch_last_inc_val = epoch
            if verbose:
                print(f"New best val !!!! -> {validation_value}")
                print(f"Saving model to {filename_best_val_model}")
            torch.save(model.state_dict(), filename_best_val_model)

            if "how_many_steps_to" in cfg:
                for thresh in how_many_steps_to.keys():
                    if validation_value > thresh and how_many_steps_to[thresh] < 0:
                        how_many_steps_to[thresh] = int(epoch * max_iter_mult_factor * cfg["max_steps_per_epoch"])
                        write_json(filename_how_many_steps_to, how_many_steps_to)


        write_json_np_ok(filename_stats_finetuning, stats_finetuning)

        if epoch - epoch_last_inc_val > cfg["max_wait_epoch_inc"]:
            break
