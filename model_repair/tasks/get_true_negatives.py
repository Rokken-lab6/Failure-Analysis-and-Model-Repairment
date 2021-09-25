from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split

from tools.utils import write_json

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def get_true_negatives(cfg, checkpoint_path, cfg_trigger, gcfg, cache):

    # NOTE: cfg_trigger is needed for cache invalidation (until better cache lib)
    cache_path = cache.get_path()

    # Load the interface
    interface = get_interface(gcfg)

    # Get the dataset
    dataloader = interface.get_dataloader(shuffle=cfg["shuffle"])

    # Get the model
    model = interface.load_model(checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.set_grad_enabled(True)

    all_tns = []

    model.eval()

    count_object = 0

    count_frames_no_tn = 0
    for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):

        
        loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)

        # Get objects -----
        objects_info = interface.get_objects(preds, sample) # List of dicts

        [e.update({"example_id": i_batch}) for e in objects_info]
        for e in objects_info:
            e.update({"object_id": count_object})
            count_object += 1

        if len(objects_info) == 0:
            assert len(sample["labels"][0][0]) == 0
            frame_name = sample["frame"][0]
            all_tns.append(frame_name)
        else:
            count_frames_no_tn += 1

        if i_batch + 1 >= cfg["n_samples"]:
            break

    print("len(all_tns) prop", len(all_tns) / count_frames_no_tn)

    train_val, test_tns = train_test_split(all_tns, test_size=0.2)
    train_tns, val_tns = train_test_split(train_val, test_size=0.2)

    model_info = {
        "proportion_tn": len(all_tns) / count_frames_no_tn,
        "train": train_tns,
        "val": val_tns,
        "test": test_tns,
    }
    path = "cached/tns_list.json"
    write_json(path, model_info)
