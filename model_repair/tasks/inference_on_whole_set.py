from tqdm import tqdm
import pandas as pd

from tools.utils import read_json

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def inference_on_whole_set(cfg, split_out, finetuned_model, split_set, gcfg, cache):

    # Load dataframe
    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info_old = pd.read_feather(str(path_df))

    # Load interface
    interface = get_interface(gcfg)

    # Get the model
    checkpoint_path = finetuned_model.get_path() / "model_best_val.pytorch"
    model = interface.load_model(checkpoint_path)

    all_objs = run(interface, model, split_set, objs_info_old, cfg)

    all_objs.to_feather(str(cache.get_path()))


def run(interface, model, split_set, objs_info_old, cfg):

    if split_set == "test":
        info = objs_info_old[(objs_info_old["split_id"]==2)]
    elif split_set == "train":
        info = objs_info_old[(objs_info_old["split_id"]==0)]
    else:
        raise NotImplementedError()
    
    example_key = interface.get_example_key()
    specific_frames = info[example_key].unique().tolist()


    # Get the dataset
    if "use_tns_list" in cfg:

        if split_set == "test":
            specific_frames += read_json(cfg["use_tns_list"])["test"]
        elif split_set == "train":
            specific_frames += read_json(cfg["use_tns_list"])["train"]
        else:
            raise NotImplementedError()
        
    if "skip_train_set" in cfg and cfg["skip_train_set"] and split_set == "train":
        skip_train_set = True
        print("skip_train_set", skip_train_set, "reducing to not many frames")
        specific_frames = specific_frames[:5]
    else:
        skip_train_set = False
        

    dataloader = interface.get_dataloader(specific_frames=specific_frames)

    all_objs = []
    for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):

        loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)
        loss = loss.cpu().detach().numpy() # Note that value of loss for the val set is different for object detection as I use grad_pos=None

        # Get the (new) objects for computing the metrics
        objs = interface.get_objects(preds, sample) # List of dicts
        objs = pd.DataFrame(objs)

        # Identify the cluster for each object (-2 means new object, -1 correct cluster)
        objs = interface.identify_cluster(objs, objs_info_old, frame_name=sample[example_key][0])

        objs["loss"] = loss

        all_objs.append(objs)

    all_objs = pd.concat(all_objs, 0).reset_index(drop=True)

    return all_objs