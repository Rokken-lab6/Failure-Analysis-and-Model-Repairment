import numpy as np
import zarr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def plot_embeddeds(cfg, split_out, embed_out, gcfg, cache):

    z_file = zarr.open(str(embed_out.get_path() / "embedded.zarr"), mode='r')

    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info = pd.read_feather(str(path_df))

    if gcfg["embed"]["only_mistakes_embedding"]:
        objs_info_f = objs_info[objs_info["correct"] == 0].reset_index()
    else:
        objs_info_f = objs_info

    embedded_2d = np.array(z_file["embedded_2d"])

    make_the_plots(objs_info_f, cfg, gcfg, embedded_2d, cache)

def make_the_plots(objs_info_f, cfg, gcfg, embedded_2d, cache):

    plots = []
    
    if "tp" in objs_info_f:
        plots.append((
            "FP_FN",
            [

                ("tp", objs_info_f[(objs_info_f["tp"]==1)]["idx_into_embeds"], "green"),
                ("tn", objs_info_f[(objs_info_f["tn"]==1)]["idx_into_embeds"], "blue"),

                ("fp", objs_info_f[(objs_info_f["fp"]==1)]["idx_into_embeds"], "orange"),
                ("fn", objs_info_f[(objs_info_f["fn"]==1)]["idx_into_embeds"], "red"),

            ]
        ))

        plots.append((
            "FP_FN_classic",
            [

                ("fp", objs_info_f[(objs_info_f["fp"]==1)]["idx_into_embeds"], "orange"),
                ("fn", objs_info_f[(objs_info_f["fn"]==1)]["idx_into_embeds"], "red"),

            ]
        ))

    if "pred" in objs_info_f:
        key = "pred"
        vals = objs_info_f[key].unique().tolist()
        vals.sort()
        n_vals = len(vals)
        colors = sns.color_palette("tab10", n_vals)

        plot_desc = []
        for i in range(n_vals):
            val = vals[i]
            color = colors[i]
            plot_desc.append((f"P{val}", objs_info_f[(objs_info_f[key]==val)]["idx_into_embeds"], color))

        plots.append(("pred_class", plot_desc))

    if "gt" in objs_info_f:
        key = "gt"
        vals = objs_info_f[key].unique().tolist()
        vals.sort()
        n_vals = len(vals)
        colors = sns.color_palette("tab10", n_vals)

        plot_desc = []
        for i in range(n_vals):
            val = vals[i]
            color = colors[i]
            plot_desc.append((f"GT{val}", objs_info_f[(objs_info_f[key]==val)]["idx_into_embeds"], color))

        plots.append(("gt_class", plot_desc))


    if "correct" in objs_info_f:
        plots.append((
            "Correct",
            [
                ("Mistake", objs_info_f[(objs_info_f["correct"]==0)]["idx_into_embeds"], "red"),
                ("Correct", objs_info_f[(objs_info_f["correct"]==1)]["idx_into_embeds"], "green"),
            ]
        ))

    if "cluster" in objs_info_f:
        cluster_ids = objs_info_f["cluster"].unique().tolist()
        cluster_ids.sort()
        n_clusters = len(cluster_ids)
        colors = sns.color_palette("tab10", n_clusters)

        clusters_plot = []
        for i in range(n_clusters):
            cluster_id = cluster_ids[i]
            if cluster_id < 0:
                continue # Ignore correct cluster in this plot
            color = colors[i]
            clusters_plot.append((f"Cluster {cluster_id+1}", objs_info_f[(objs_info_f["cluster"]==cluster_id)]["idx_into_embeds"], color))

        plots.append(("Cluster", clusters_plot))

    if "split_id" in objs_info_f:

        # Only plot mistakes for this plot
        objs_info_f_only_mistakes = objs_info_f[(objs_info_f["correct"]==0)]

        vals = objs_info_f_only_mistakes["split_id"].unique().tolist()
        vals.sort()
        n_vals = len(vals)
        colors = sns.color_palette("tab10", n_vals)

        plot_desc = []
        for i in range(n_vals):
            val = vals[i]
            color = colors[i]
            plot_desc.append((f"Split {val}", objs_info_f_only_mistakes[(objs_info_f_only_mistakes["split_id"]==val)]["idx_into_embeds"], color))

        plots.append(("Split", plot_desc))

    if "label_group" in objs_info_f:
        key = "label_group"
        vals = objs_info_f[key].unique().tolist()
        vals.sort()
        n_vals = len(vals)
        colors = sns.color_palette("tab10", n_vals)

        plot_desc = []
        for i in range(n_vals):
            val = vals[i]
            color = colors[i]
            plot_desc.append((f"Label {val}", objs_info_f[(objs_info_f[key]==val)]["idx_into_embeds"], color))

        plots.append(("Original label", plot_desc))

    if "label_is_wrong" in objs_info_f:
        key = "label_is_wrong"
        vals = objs_info_f[key].unique().tolist()
        vals.sort()
        n_vals = len(vals)
        colors = sns.color_palette("tab10", n_vals)
 
        plot_desc = []
        for i in range(n_vals):
            val = vals[i]
            color = colors[i]
            plot_desc.append((f"Label {val}", objs_info_f[(objs_info_f[key]==val)]["idx_into_embeds"], color))

        plots.append(("Label is wrong", plot_desc))


    # Cluster plot with marker shapes for some key
    if "cluster" in objs_info_f and "key_for_markers" in cfg and cfg["key_for_markers"]:

        if "show_legend" in cfg:
            show_legend = cfg["show_legend"]
        else:
            show_legend = True

        cluster_ids = objs_info_f["cluster"].unique().tolist()
        cluster_ids.sort()
        n_clusters = len(cluster_ids)
        colors = sns.color_palette("tab10", n_clusters)

        values_for_markers = objs_info_f[cfg["key_for_markers"]].unique().tolist()
        values_for_markers.sort()


        clusters_plot = []
        for i in range(n_clusters):
            cluster_id = cluster_ids[i]
            if cluster_id < 0:
                continue # Ignore correct cluster in this plot
            
            color = colors[i]

            for j in range(len(values_for_markers)):
                value_marker = values_for_markers[j]
                marker = cfg.markers[j]
                key_marker = cfg["key_for_markers"]


                idx = objs_info_f[(objs_info_f["cluster"]==cluster_id) & (objs_info_f[cfg["key_for_markers"]]==value_marker)]["idx_into_embeds"]


                if key_marker == "gt" or key_marker == "fn":
                    key_marker = ""
                    if value_marker == 0:
                        value_marker = "fp"
                    elif value_marker == 1:
                        value_marker = "fn"
                    text_label = f"{value_marker}"
                elif key_marker == "fp":
                    key_marker = ""
                    if value_marker == 0:
                        value_marker = "fn"
                    elif value_marker == 1:
                        value_marker = "fp"
                    text_label = f"{value_marker}"
                else:
                    text_label = f"{key_marker}:{value_marker}"

                clusters_plot.append((f"Cluster {cluster_id+1} {text_label}", idx, color, marker, show_legend))

        plots.append((f"Cluster marker {cfg['key_for_markers']}", clusters_plot))

    if gcfg["trigger_extract"].epochs is None or len(gcfg["trigger_extract"].epochs) > 1:
        # If 1 epoch, it can be the ref epoch..
        epoch = int(gcfg["trigger_extract"]["ref_epoch"])
    elif len(gcfg["trigger_extract"].epochs) == 1:
        epoch = gcfg["trigger_extract"].epochs[0]

    layer_name = "all" if gcfg["embed"]["layers"] is None else gcfg["embed"]["layers"]

    for plot in plots:
        plot_name = plot[0]
        categories = plot[1]
        
        path_out = cache.get_path() / plot_name / f"{epoch}_{layer_name}.png"
        path_out.parent.mkdir(parents=True, exist_ok=True)

        plot_scatter(embedded_2d, categories, scale=20, alpha=0.3, title=f"Embedding, \n {plot_name}, layers: {layer_name}, epoch: {epoch}", path_save=str(path_out))
        
def plot_scatter(umap_embed_2d, categories, scale=10, alpha=0.9, title="Title", path_save=None):
    fig, ax = plt.subplots()

    for category in categories:
        points = umap_embed_2d[category[1]]
        if len(category) >= 4:
            marker = category[3]
            show_legend = category[4]
        else:
            marker = "o"
            show_legend = True
        ax.scatter(points[:, 0], points[:, 1], color=category[2], s=scale, label=category[0],
                   alpha=alpha, edgecolors='none', marker=marker)

    if show_legend:
        ax.legend()
    plt.title(title, fontsize=16)
    fig.tight_layout()
    if path_save is not None:
        plt.savefig(path_save, dpi=300)
    plt.close()