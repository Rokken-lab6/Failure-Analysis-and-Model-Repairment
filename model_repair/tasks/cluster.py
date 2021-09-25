import numpy as np
import zarr
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def cluster(cfg, extracted, embed_out, gcfg, cache):

    # Load embedding data
    z_file_embeds = zarr.open(str(embed_out.get_path() / "embedded.zarr"), mode='r')
    embedded_nd = np.array(z_file_embeds["embedded_nd"])

    # Load objs_info
    path_per_obj_info = extracted.get_path() / "per_object_info.feather"
    objs_info = pd.read_feather(str(path_per_obj_info))

    # Filter data to only mistakes    
    if gcfg["embed"]["only_mistakes_embedding"]:

        # Add new column with proper index in embedded_nd (if exists)
        mistakes = (objs_info["correct"] == 0).to_numpy().astype(int)
        idx_into_embeds = []
        count = 0
        for idx in range(mistakes.shape[0]):
            if mistakes[idx] > 0.5:
                idx_into_embeds.append(count)
                count += 1
            else:
                idx_into_embeds.append(-1)
        objs_info["idx_into_embeds"] = idx_into_embeds

        # Only mistakes were embedded so need to filter objs_info to only mistakes
        objs_info_f = objs_info[objs_info["correct"] == 0].reset_index()
        objs_info_correct = objs_info[objs_info["correct"] == 1].reset_index()
    else:
        # Non mistakes were also embedded to need to filter non-mistakes out of both objs_info and embedded_nd

        objs_info["idx_into_embeds"] = list(range(embedded_nd.shape[0])) # TODO: Check
        
        embedded_nd = embedded_nd[objs_info["correct"] == 0]
        objs_info_f = objs_info[objs_info["correct"] == 0].reset_index()
        objs_info_correct = objs_info[objs_info["correct"] == 1].reset_index()

    # Cluster (using max silhouette score)
    if cfg["algorithm"] == "random":
        
        N_points = objs_info_f.shape[0]
        best_labels = np.random.randint(cfg["n_random_clusters"], size=N_points)
        best_centers = embedded_nd[:cfg["n_random_clusters"]] # NOTE: Just pick the first ones ... (no meaning) # TODO: Use None instead ?

    elif cfg["algorithm"] == "metadata":


        if "sort_by_count" in cfg and cfg["sort_by_count"]:
            cluster_labels = objs_info_f[cfg.metadata_key].value_counts().keys().tolist()
        else:
            cluster_labels = objs_info_f[cfg.metadata_key].unique()
            cluster_labels.sort()
            cluster_labels = list(cluster_labels)

        if "fix_small_metadata_cluster" in cfg and cfg["fix_small_metadata_cluster"]:
            to_fix = []
            for meta_value, count in objs_info_f[cfg.metadata_key].value_counts().iteritems():
                if count < 3:
                    print(f"WARNING: meta label {meta_value} has count {count}, putting aside")
                    cluster_labels.remove(meta_value)
                    to_fix.append(meta_value)

        map_ = {k: i for i, k in enumerate(cluster_labels)}

        if "fix_small_metadata_cluster" in cfg and cfg["fix_small_metadata_cluster"]:
            for e in to_fix:
                map_[e] = -3

        n_clusters = len(cluster_labels)
        N_points = objs_info_f.shape[0]
        
        objs_info_f["best_labels"] = objs_info_f[cfg.metadata_key].apply(lambda e: map_[e])
        best_labels = objs_info_f["best_labels"]


        best_centers = []
        for cluster_id in range(n_clusters):
            idx = int(objs_info_f[objs_info_f["best_labels"]==cluster_id].iloc[0].name)
            best_centers.append(embedded_nd[idx])
        best_centers = np.array(best_centers)
    else:
        best_labels, best_centers = do_clustering(embedded_nd, cfg)

    # Save a new feather dataframe containing the clustering information ------------------------------------------------
    objs_info_f["cluster"] = best_labels
    objs_info_correct["cluster"] = -1

    # Merge clustered and non clustered(correct) objects into one dataframe
    new_obj_info = pd.concat([objs_info_f, objs_info_correct], 0).reset_index(drop=True)

    # Normaly this should order the dataframe by object_id and keep the same order as originally
    new_obj_info = new_obj_info.sort_values("object_id").reset_index(drop=True)

    assert new_obj_info[new_obj_info["cluster"]==0]["correct"].max() == 0

    print("Saving df with clusters info ...")

    out_path = cache.get_path()
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(new_obj_info)
    path = out_path / "clustered.feather"
    df.to_feather(str(path))

    # Saving best_centers info
    with open(str(out_path / "best_centers.joblib"), 'wb') as f:
        joblib.dump(best_centers, f)

def do_clustering(embedded_nd, cfg):

    sil_score_max = -1 # This is the minimum possible score

    for n_clusters in range(cfg["min_clusters"], cfg["max_clusters"]):
        
        if cfg["algorithm"] == "k-means":
            model = KMeans(n_clusters=n_clusters, init='k-means++')
        else:
            raise NotImplementedError()

        labels = model.fit_predict(embedded_nd)
        
        sil_score = silhouette_score(embedded_nd, labels)
        
        print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
                
        if sil_score >= sil_score_max:
            sil_score_max = sil_score
            best_labels = labels
            best_centers = model.cluster_centers_

    return best_labels, best_centers
