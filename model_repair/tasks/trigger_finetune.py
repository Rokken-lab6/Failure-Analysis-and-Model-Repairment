import pandas as pd
from tools.utils import read_json, write_json

from model_repair.cache42.cached_42 import cache_42

from .finetune_on_cluster import finetune_on_cluster
from .finetune_on_cluster_ewc import finetune_on_cluster_ewc
from .get_another_epoch import get_another_epoch
from .inference_on_whole_set import inference_on_whole_set
from .evaluate_on_all_clusters import evaluate_on_all_clusters

@cache_42(ignore_args=["gcfg"], force_recompute=True)
# @cache_42(ignore_args=["gcfg"])
def trigger_finetune(cfg, split_out, gcfg, cache):

    # Load dataframe
    path_df = split_out.get_path() / "clustered_and_splitted.feather"
    objs_info = pd.read_feather(str(path_df))

    n_clusters = objs_info["cluster"].max() + 1

    finetuneds = {}

    if cfg.use_ewc_finetune:
        finetune_task = finetune_on_cluster_ewc
        finetune_cfg_key = "finetune_on_cluster_ewc"
    else:
        finetune_task = finetune_on_cluster
        finetune_cfg_key = "finetune_on_cluster"
    
    verbose = True
    if "skip_clusters" in cfg and cfg["skip_clusters"]:
        skip_clusters = True
    else:
        skip_clusters = False

    if "skip_finetune_all_nothing_other" in cfg and cfg["skip_finetune_all_nothing_other"]:
        skip_finetune_all_nothing_other = True
    else:
        skip_finetune_all_nothing_other = False

    # Launch finetuning on each cluster
    if not skip_clusters:
        for cluster_id in range(n_clusters):
            res = finetune_task(cfg[finetune_cfg_key], split_out, cluster_id, verbose, gcfg=gcfg)
            finetuneds[cluster_id] = res

    # Finetune on all clusters at once
    if not skip_finetune_all_nothing_other:
        finetuneds["all"] = finetune_task(cfg[finetune_cfg_key], split_out, cluster_id="all", verbose=verbose, gcfg=gcfg)

    # Don't finetune (create a res object with a non finetuned model)
    if not skip_finetune_all_nothing_other:
        finetuneds["nothing"] = finetune_task(cfg[finetune_cfg_key], split_out, cluster_id="nothing", verbose=verbose, gcfg=gcfg)

    # Don't finetune and use another epoch
    if not skip_finetune_all_nothing_other:
        finetuneds["other_epoch"] = get_another_epoch(cfg["get_another_epoch"], split_out, gcfg=gcfg)

    metrics_test = {}
    metrics_train = {}
    stats_finetuning = {}
    for finetuning_cluster_id in finetuneds:
        print(f"@@@@@@@@@@@@@@@@@@@@ Evaluating model finetuned on {finetuning_cluster_id}...")

        finetuned = finetuneds[finetuning_cluster_id]

        # Test
        print("Test set...")
        inferenced = inference_on_whole_set(cfg["inference_on_whole_set"], split_out, finetuned, "test", gcfg=gcfg)
        res_test = evaluate_on_all_clusters(cfg["evaluate_on_all_clusters"], split_out, inferenced, "test", gcfg=gcfg)
        metrics_test[finetuning_cluster_id] = read_json(res_test.get_path())

        # Train
        print("Train set...")
        inferenced = inference_on_whole_set(cfg["inference_on_whole_set"], split_out, finetuned, "train", gcfg=gcfg)
        res_train = evaluate_on_all_clusters(cfg["evaluate_on_all_clusters"], split_out, inferenced, "train", gcfg=gcfg)
        metrics_train[finetuning_cluster_id] = read_json(res_train.get_path())

        try:
            stats_finetuning[finetuning_cluster_id] = read_json(finetuned.get_path() / "stats_finetuning.json")
        except:
            stats_finetuning[finetuning_cluster_id] = None

    metrics = {
        "train": metrics_train,
        "test": metrics_test,
        "stats_finetuning": stats_finetuning,
    }
    write_json(cache.get_path(), metrics)

    return cache