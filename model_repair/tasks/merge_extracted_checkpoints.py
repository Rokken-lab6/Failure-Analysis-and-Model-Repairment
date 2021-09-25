from tqdm import tqdm
import numpy as np
import zarr
import pandas as pd

from tools.utils import write_json, read_json

def merge_extracted_checkpoints(extracteds, ref_extracted, out_path):

    data_grads = []
    data_feats = []
    data_featsavg = []
    data_correct = []
    for extracted in extracteds:
        
        z_file = zarr.open(str(extracted.get_path() / "extracted.zarr"), mode='r')
        path_per_obj_info = extracted.get_path() / "per_object_info.feather"
        objs_info = pd.read_feather(str(path_per_obj_info))

        data_grads_i = get_data(z_file, key="per_object_grads")
        data_grads.append(data_grads_i)
        
        data_feats_i = get_data(z_file, key="images_feats_normal")
        data_feats.append(data_feats_i)

        data_featsavg_i = get_data(z_file, key="images_feats_avg")
        data_featsavg.append(data_featsavg_i)

        data_correct_i = objs_info["correct"].astype(int).to_numpy()
        data_correct.append(data_correct_i)

    # Merge grads
    # data_grads of shape [n_checkpoints, n_levels, n_images, features]
    n_checkpoints = len(data_grads)
    n_levels = len(data_grads[0])
    merged_grads = []
    for level in range(n_levels):
        yo = np.concatenate([data_grads[c][level] for c in range(n_checkpoints)], -1)
        merged_grads.append(yo)

    # Merge features_normal
    # features_normal of shape [n_checkpoints, n_levels, n_images, features]
    n_checkpoints = len(data_feats)
    n_levels = len(data_feats[0])
    merged_data_feats = []
    for level in range(n_levels):
        yo = np.concatenate([data_feats[c][level] for c in range(n_checkpoints)], -1)
        merged_data_feats.append(yo)

    # Merge data_featsavg
    # data_featsavg of shape [n_checkpoints, n_levels, n_images, features]
    n_checkpoints = len(data_featsavg)
    n_levels = len(data_featsavg[0])
    merged_data_featsavg = []
    for level in range(n_levels):
        yo = np.concatenate([data_featsavg[c][level] for c in range(n_checkpoints)], -1)
        merged_data_featsavg.append(yo)

    all_objects_info = pd.read_feather(str(ref_extracted.get_path() / "per_object_info.feather"))
    
    model_info = read_json(str(ref_extracted.get_path() / "model_info.json"))
    feats_layers_info = model_info["feats_layers_info"]
    grads_layers_info = model_info["grads_layers_info"]

    save_data(merged_grads, merged_data_feats, merged_data_featsavg, all_objects_info, feats_layers_info, grads_layers_info, data_correct, out_path)

def get_data(z_file, key):

    n_levels = get_n_levels(z_file, key=key)
    layers_ids = range(n_levels)
    data = []
    for gi in layers_ids:
        
        data_gi = z_file[f"{key}/{gi}"]

        data.append(data_gi)

    return data

def get_n_levels(z_file, key):
    for i in range(10000):
        try:
            len(z_file[f"{key}/{i}"])
        except:
            return i

def save_data(all_objects_grads, all_objects_features_normal, all_objects_features_avg, all_objects_info, feats_layers_info, grads_layers_info, data_correct, out_path):

    out_path.mkdir(exist_ok=True, parents=True)

    path = str(out_path / "extracted.zarr")
    z_file = zarr.open(path, mode='w')

    print("Saving per object gradients...")
    z_per_object_grads = z_file.create_group("per_object_grads")
    N_levels = len(all_objects_grads) # Note: Opposite than in extract_checkpoint
    if N_levels > 0:
        for li in tqdm(range(N_levels)):

            # Stack all objects for level li
            yo = all_objects_grads[li]
            z_li = z_per_object_grads.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No grad information available")
    print("Saving images_feats_normal...")
    z_feats = z_file.create_group("images_feats_normal")
    N_levels = len(all_objects_features_normal)
    if N_levels > 0:
        
        for li in tqdm(range(N_levels)):

            yo = all_objects_features_normal[li]
            z_li = z_feats.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No feature information available")

    print("Saving images_feats_avg...")
    z_feats = z_file.create_group("images_feats_avg")
    N_levels = len(all_objects_features_avg)
    if N_levels > 0:
        
        for li in tqdm(range(N_levels)):

            yo = all_objects_features_avg[li]

            z_li = z_feats.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No feature information available")


    print("Saving correct...")
    z_correct = z_file.create_group("correct")
    li = 0
    yo = np.array(data_correct).T # [N, n checkpoints]
    z_li = z_correct.zeros(str(li), shape=yo.shape)
    z_li[:] = yo[:]

    print("Saving per object info...")
    df = pd.DataFrame(all_objects_info)
    path = out_path / "per_object_info.feather"
    df.to_feather(str(path))

    print("Saving model_info...")
    model_info = {
        "feats_layers_info": feats_layers_info,
        "grads_layers_info": grads_layers_info,
    }
    write_json(str(out_path / "model_info.json"), model_info)
