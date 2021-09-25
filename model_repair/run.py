#!/usr/bin/env python

import argparse
from pathlib import Path
import shutil
from datetime import datetime

from box import Box

import tools.pudb_hook
from .tools.utils import read_yaml

from model_repair.cache42 import cached_42

from .tasks.trigger_extract import trigger_extract
from .tasks.embed import embed
from .tasks.plot_embeddeds import plot_embeddeds
from .tasks.cluster import cluster
from .tasks.split_clusters import split_clusters
from .tasks.show_clusters_content import show_clusters_content
from .tasks.make_cluster_stats import make_cluster_stats
from .tasks.trigger_finetune import trigger_finetune
from .tasks.plot_heatmaps import plot_heatmaps
from .tasks.plot_finetuning_curves import plot_finetuning_curves
from .tasks.compute_cluster_metrics_bounded import compute_cluster_metrics_bounded


def run_pipeline(cfg_path):

    cfg = Box(read_yaml(cfg_path))

    cached_42.STORAGE = cfg.pipeline.storage

    # Extract feature/gradient representation of each failure case.
    extracted_out = trigger_extract(cfg["trigger_extract"], gcfg=cfg)

    # Embed using UMAP
    embed_out = embed(cfg["embed"], extracted_out)

    # Cluster the failures in the embedded space
    cluster_out = cluster(cfg["cluster"], extracted_out, embed_out, gcfg=cfg)
    
    # Split each cluster into a training, validation and testing subsets.
    split_out = split_clusters(cfg["split_clusters"], cluster_out, gcfg=cfg)

    # Plot the embeddings
    embed_plots = plot_embeddeds(cfg["plot_embeddeds"], split_out, embed_out, gcfg=cfg)

    # Show the content of each cluster
    cluster_contents = show_clusters_content(cfg["show_clusters_content"], embed_out, cluster_out, gcfg=cfg)

    # Show some statistics on the content of each cluster
    cluster_stats = make_cluster_stats(cfg["make_cluster_stats"], embed_out, cluster_out, split_out, gcfg=cfg)

    # Finetune on each cluster first independently then all at once and finally evaluate the model on all.
    finetuned_metrics = trigger_finetune(cfg["trigger_finetune"], split_out, gcfg=cfg)

    # Plot the finetuning matrices
    heatmaps = plot_heatmaps(cfg["plot_heatmaps"], finetuned_metrics, gcfg=cfg)

    # Compute the surrogate metrics: Learnability and Independence
    cluster_metrics_bounded = compute_cluster_metrics_bounded(cfg["compute_cluster_metrics_bounded"], finetuned_metrics, gcfg=cfg)
    
    if "plot_finetuning_curves" not in cfg:
        cfg["plot_finetuning_curves"] = {}

    finetuning_curves = plot_finetuning_curves(cfg["plot_finetuning_curves"], finetuned_metrics, gcfg=cfg)

    # Copy the results (possibly cached) to the "out/report" folder
    report(cfg, [
        embed_plots,
        cluster_contents,
        cluster_stats,
        heatmaps,
        cluster_metrics_bounded,
        finetuning_curves,
    ])

def report(cfg, to_report):

    flow_name = cfg["report"]["flow_name"]
    target = Path(cfg["report"].path) / flow_name
    
    # If folder exists, just add the time to the name 
    if target.exists():
        current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        target = Path(cfg["report"].path) / f"{flow_name}_{current_time_str}"

    target.mkdir(parents=True, exist_ok=True)

    print(f"Reporting to {target} ")

    for cache in to_report:
        task_name = Path(cache.get_path()).parent.stem
        if Path(cache.get_path()).is_dir():
            shutil.copytree(cache.get_path(), target / task_name)
        elif Path(cache.get_path()).exists():
            shutil.copy2(cache.get_path(), target / task_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='cfg', type=str, required=True)
    args = parser.parse_args()

    run_pipeline(args.cfg)