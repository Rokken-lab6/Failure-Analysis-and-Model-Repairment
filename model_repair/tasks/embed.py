import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import zarr
import pandas as pd
import torch
import umap

import tools.pudb_hook
from tools.utils import get_class, write_json

from model_repair.cache42.cached_42 import cache_42

# @cache_42(force_recompute=True)
@cache_42()
def embed(cfg, extracted, cache):

    z_file = zarr.open(str(extracted.get_path() / "extracted.zarr"), mode='r')

    if cfg["embedding_features"] == "grads":
        data = get_data(z_file, cfg["layers"], key="per_object_grads")
    elif cfg["embedding_features"] == "feats_normal":
        data = get_data(z_file, cfg["layers"], key="images_feats_normal")
    elif cfg["embedding_features"] == "feats_avg":
        data = get_data(z_file, cfg["layers"], key="images_feats_avg")
    elif cfg["embedding_features"] == "correct":
        data = get_data(z_file, [0], key="correct")

    path_per_obj_info = extracted.get_path() / "per_object_info.feather"
    objs_info = pd.read_feather(str(path_per_obj_info))

    assert data.shape[0] == len(objs_info) # 1 entry per object in both

    if cfg["only_mistakes_embedding"]:
        mask_ids = objs_info[objs_info["correct"] == 0]["object_id"].tolist()
        data_f = data[mask_ids]
    else:
        data_f = data

    assert cfg["method"] == "umap"
    if cfg["normalize"]:
        print("Normalizing StandardScaler...")
        from sklearn.preprocessing import normalize
        from sklearn import preprocessing

        scaler = preprocessing.StandardScaler().fit(data_f)
        data_f = scaler.transform(data_f)

    print("Embedding nd...")
    embedded_nd = umap_embed(data_f, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"], n_components=cfg["ND"], metric='euclidean', low_memory=True)

    print("Embedding 2d...")
    embedded_2d = umap_embed(data_f, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"], n_components=2, metric='euclidean', low_memory=True)

    cache_path = cache.get_path()

    # Save to zarr format
    cache_path.mkdir(exist_ok=True, parents=True)

    z_file_out = zarr.open(str(cache_path / "embedded.zarr"), mode='w')

    print("Saving embedded_nd...")
    z_li = z_file_out.zeros("embedded_nd", shape=embedded_nd.shape)
    z_li[:] = embedded_nd[:]

    print("Saving embedded_2d...")
    z_li = z_file_out.zeros("embedded_2d", shape=embedded_2d.shape)
    z_li[:] = embedded_2d[:]

def get_data(z_file, layers_ids, key):

    if layers_ids is None:
        n_levels = get_n_levels(z_file, key=key)
        layers_ids = range(n_levels)

    data = []
    for gi in layers_ids:
        
        data_gi = z_file[f"{key}/{gi}"]

        data.append(data_gi)
    data = np.concatenate(data, 1)
    
    return data

def get_n_levels(z_file, key):
    for i in range(10000):
        try:
            len(z_file[f"{key}/{i}"])
        except:
            return i

def umap_embed(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', low_memory=True):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory
    )

    u = fit.fit_transform(data)
    return u