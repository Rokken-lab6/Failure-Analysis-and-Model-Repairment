from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def make_cluster_stats(cfg, embed_out, cluster_out, split_out, gcfg, cache):
    print("make_cluster_stats...")

    # Load dataframe
    path_df = cluster_out.get_path() / "clustered.feather"
    objs_info = pd.read_feather(str(path_df))

    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info_splitted = pd.read_feather(str(path_df))

    interface = get_interface(gcfg)

    stats_keys = interface.get_cluster_stats_keys()

    # Per cluster stats ---------------------------------------------------------
    out_path_stats = Path(cache.get_path()) / "stats"
    
    plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column="split_id", prefix="")
    plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column="correct", prefix="")

    for key in stats_keys["keys"]:
        if key in objs_info_splitted.keys().tolist():
            plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column=key, prefix="")

    # Per cluster stats (no -1 cluster) ---------------------------------------------------------
    out_path_stats = Path(cache.get_path()) / "stats_no_m_1"
    
    plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column="split_id", prefix="", only_mistakes=True)
    plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column="correct", prefix="", only_mistakes=True)

    for key in stats_keys["keys"]:
        if key in objs_info_splitted.keys().tolist():
            plot_count_cm(objs_info_splitted, out_path_stats, key_row="cluster", key_column=key, prefix="", only_mistakes=True)


    # Cross keys stats per cluster ----------------------------------------------
    cluster_ids = objs_info["cluster"].unique()
    cluster_ids.sort() # Sort from -2, -1 to n_clusters

    for k_row, k_col in stats_keys["cross_keys"]:

        out_path_stats = Path(cache.get_path()) / "cross" / f"{k_row}x{k_col}"

        all_stats = []
        for cluster_id in cluster_ids:
            stats = plot_count_cm(objs_info_splitted, out_path_stats, key_row=k_row, key_column=k_col, prefix=f"c{cluster_id}_", cluster_id=cluster_id)
            
            if cluster_id >= 0: # Only mistake clusters
                all_stats.append(stats)

        all_stats = np.stack(all_stats, -1)

        proportion_per_cross_cell_for_clusters = all_stats/all_stats.sum(-1, keepdims=True)
        plot_count_cm_overlap(proportion_per_cross_cell_for_clusters, out_path_stats)

    return cache


def plot_count_cm(objs_info, out_path, key_row, key_column, prefix="", cluster_id=None, only_mistakes=False):
    pass

    out_path.mkdir(parents=True, exist_ok=True)

    if only_mistakes:
        objs_info = objs_info[objs_info.cluster >=0]

    all_rows = objs_info[key_row].unique()
    all_rows.sort()

    all_cols = objs_info[key_column].unique()
    all_cols = [e if e is not None else "Unknown" for e in all_cols]
    all_cols.sort()

    if cluster_id is not None:
        objs_info = objs_info[objs_info["cluster"]==cluster_id]
    
    stats = objs_info.pivot_table(index=[key_row], columns=key_column, aggfunc='size', fill_value=0)

    stats = stats.reindex(all_cols, axis=1, fill_value=0).reindex(all_rows, axis=0, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10,10)) 
    cmap = "cividis"

    if key_row == "cluster" and only_mistakes:
        # Clusters number from 1
        yticklabels = list(all_rows + 1)
        stats = stats.div(stats.sum(axis=1), axis=0) # Normalize by row
        ax = sns.heatmap(stats, annot=True, cmap=cmap, square=True, annot_kws={"fontsize": 12}, ax=ax, yticklabels=yticklabels)
    else:
        ax = sns.heatmap(stats, annot=True, cmap=cmap, square=True, annot_kws={"fontsize": 12}, ax=ax)


    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 8)

    plt.savefig(str(out_path / (prefix + f"{key_row}X{key_column}.png")), dpi=400)
    plt.close()

    return stats

def plot_count_cm_overlap(proportion_per_cross_cell_for_clusters, out_path_stats):

    H, W, C = proportion_per_cross_cell_for_clusters.shape
    size = 10
    sizep1 = size + 1

    colors = sns.color_palette("tab10", proportion_per_cross_cell_for_clusters.shape[-1])

    img = np.zeros((H*sizep1, W*sizep1, 3))

    for i in range(H):
        for j in range(W):
            img_cell = make_one_cell_overlap(proportion_per_cross_cell_for_clusters[i, j], size=size, colors=colors)
            img[i*sizep1:(i+1)*sizep1, j*sizep1:(j+1)*sizep1] = img_cell

    plt.imshow(img)

    plt.savefig(str(out_path_stats / "cross_proportions.png"))
    plt.close()


def make_one_cell_overlap(proportions, size, colors):

    img_cell = np.zeros((size*size, 3))

    if not np.isnan(proportions.max()):
        sorted_idx = (-proportions).argsort()

        linspace = np.linspace(0,1,size*size)
        cumsum = proportions[sorted_idx].cumsum()
        cumsum = np.array([e if e < 0.9999 else 1.0 for e in cumsum ]) # Solve cumsum doesn't go to exactly 1 (float issues)
        for i, idx in enumerate(sorted_idx):
            color = colors[sorted_idx[i]] # Is that correct ?

            if i > 0:
                mask = (cumsum[i - 1] < linspace) & (linspace <= cumsum[i])
            else:
                mask = linspace <= cumsum[i]

            img_cell[mask] = color
    
    img_cell = img_cell.reshape(size, size, 3)
    img_cell = np.pad(img_cell, [(0, 1), (0, 1), (0, 0)], mode='constant', constant_values=1)
    
    return img_cell