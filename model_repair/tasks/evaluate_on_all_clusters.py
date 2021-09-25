import pandas as pd

from model_repair.cache42.cached_42 import cache_42

from tools.utils import write_json

from model_repair.override_for_release import get_interface

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def evaluate_on_all_clusters(cfg, split_out, inferenced, split_set, gcfg, cache):

    # Load dataframe old objects
    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info_old = pd.read_feather(str(path_df))

    # Load dataframe objects after finetuning
    objs = pd.read_feather(str(inferenced.get_path()))

    # Load interface to get split algorithm adapted to the task
    interface = get_interface(gcfg)

    n_clusters = objs_info_old["cluster"].max() + 1
    metrics_saved = {}
    
    for cluster_id in range(-2, n_clusters):

        objs_f = objs[objs["cluster"] == cluster_id]
        m = interface.compute_metrics(objs_f)
        m["count"] = len(objs_f)
        metrics_saved[f"cluster_{cluster_id}"] = m


    # Outside cluster (-1 and -2)
    cluster_id = "outside"
    # Outside cluster is -1 (orignally correct) and -2 (new objects)
    objs_f = objs[objs["cluster"] < 0]
    m = interface.compute_metrics(objs_f)
    m["count"] = len(objs_f)
    metrics_saved[f"cluster_outside"] = m


    m = interface.compute_metrics(objs)
    m["count"] = len(objs)
    metrics_saved[f"cluster_all"] = m

    # Save metrics_saved to disk
    write_json(cache.get_path(), metrics_saved)

    return cache