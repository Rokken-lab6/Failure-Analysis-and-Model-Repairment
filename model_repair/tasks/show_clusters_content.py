
from pathlib import Path

from tqdm import tqdm
import numpy as np
import zarr
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def show_clusters_content(cfg, embed_out, cluster_out, gcfg, cache):

    # Load dataframe
    path_df = cluster_out.get_path() / "clustered.feather"
    objs_info = pd.read_feather(str(path_df))

    best_centers = joblib.load(str(cluster_out.get_path() / "best_centers.joblib"))

    # Load embedding data
    z_file_embeds = zarr.open(str(embed_out.get_path() / "embedded.zarr"), mode='r')
    embedded_nd = np.array(z_file_embeds["embedded_nd"])

    interface = get_interface(gcfg)

    n_clusters = objs_info["cluster"].unique().max() + 1

    assert objs_info[objs_info["cluster"] >= 0]["idx_into_embeds"].max() < embedded_nd.shape[0]

    out_path_images = Path(cache.get_path()) / "images"
    out_path_plot = Path(cache.get_path()) / "plots"

    for cluster_id in tqdm(range(n_clusters)):
        objs_f = objs_info[objs_info["cluster"]==cluster_id]

        embedded_nd_f = embedded_nd[objs_f["idx_into_embeds"]]

        if cfg["n_examples"] > len(objs_f):
            how_many = len(objs_f)
        else:
            how_many = cfg["n_examples"]

        if "get_nearest" not in cfg or cfg["get_nearest"]:
            neigh = NearestNeighbors(n_neighbors=how_many)
            neigh.fit(embedded_nd_f)
            dists, found_ids = neigh.kneighbors(best_centers[np.newaxis, cluster_id], how_many, return_distance=True)
            assert dists.shape[0] == 1 # Just in case
            dists, found_ids = dists[0], found_ids[0] # Remove first dim
            N_found = dists.shape[0]
        else:
            # Pick examples at random
            N_found = min(embedded_nd_f.shape[0], how_many)
            dists = [0 for e in range(embedded_nd_f.shape[0])]
            found_ids = np.random.choice(embedded_nd_f.shape[0], how_many)

        show_objs = []
        show_dists = []
        for i in range(N_found):
            interface.export_datapoint(objs_f.iloc[found_ids[i]], dists[i], i, out_path_images / f"{cluster_id}")

            show_objs.append(objs_f.iloc[found_ids[i]])
            show_dists.append(dists[i])


        interface.plot_cluster_description(show_objs, show_dists, out_path_plot / f"{cluster_id}.png")

    return cache
