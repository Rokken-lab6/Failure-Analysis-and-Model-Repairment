import pandas as pd

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def split_clusters(cfg, cluster_out, gcfg, cache):

    # Load dataframe
    path_df = cluster_out.get_path() / "clustered.feather"
    objs_info = pd.read_feather(str(path_df))

    # Load interface to get split algorithm adapted to the task
    interface = get_interface(gcfg)

    # Split
    n_clusters = objs_info["cluster"].max() + 1
    
    all_dfs = []
    for cluster_id in range(-1, n_clusters): # Correct cluster is also split
        df_cluster = objs_info[objs_info["cluster"]==cluster_id]
        df_cluster_splited = interface.split_cluster(cfg, df_cluster)

        all_dfs.append(df_cluster_splited)

    # Merge all_dfs
    objs_info_splitted = pd.concat(all_dfs, 0).reset_index(drop=True)

    # Order the dataframe by object_id and keep the same order as originally
    objs_info_splitted = objs_info_splitted.sort_values("object_id").reset_index(drop=True)

    out_path = cache.get_path()
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(objs_info_splitted)
    path = out_path / "clustered_and_splitted.feather"
    df.to_feather(str(path))