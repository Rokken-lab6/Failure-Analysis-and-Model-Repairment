from pathlib import Path

import numpy as np

from prettytable import PrettyTable

from tools.utils import read_json, write_json

from model_repair.cache42.cached_42 import cache_42

# In the MICCAI2021 paper:
# Independence (paper) = I3 (this code)
# Learnability (paper) = learnability (this code)

@cache_42(ignore_args=["gcfg"], force_recompute=True)
# @cache_42(ignore_args=["gcfg"])
def compute_cluster_metrics_bounded(cfg, finetuned_metrics, gcfg, cache):

    metrics = read_json(finetuned_metrics.get_path())

    # Automatically get all the metrics that exists from the dict
    finetuning_groups = list(metrics["test"].keys())
    eval_groups = list(metrics["test"][finetuning_groups[0]].keys())
    metrics_keys = list(metrics["test"][finetuning_groups[0]][eval_groups[0]].keys())

    metrics_keys = [k for k in metrics_keys if k not in ["tp", "fp", "tn", "fn", "count", "loss"]]

    cluster_metrics = {}
    for metric_key in metrics_keys:
        cluster_metrics[metric_key] = {}

        # Intra
        gen_metric, learn_metric, lg_metric = compute_intra_metrics(metrics, finetuning_groups, eval_groups, metric_key)
        cluster_metrics[metric_key]["generalizability"] = gen_metric
        cluster_metrics[metric_key]["learnability"] = learn_metric # <- Used in the paper for "Learnability"
        cluster_metrics[metric_key]["lg"] = lg_metric

        # Inter (old metric)
        res = compute_inter_metric(metrics, finetuning_groups, eval_groups, metric_key)
        cluster_metrics[metric_key]["independence"] = res
    
        # Compatibility
        res = compute_compatibility(metrics, finetuning_groups, eval_groups, metric_key)
        cluster_metrics[metric_key]["compatibility"] = res

        # Alternatives for inter
        res = compute_inter_metric_I2(metrics, finetuning_groups, eval_groups, metric_key)
        cluster_metrics[metric_key]["independence_I2"] = res

        res = compute_inter_metric_I3(metrics, finetuning_groups, eval_groups, metric_key)  # <- Used in the paper for "Independence"
        cluster_metrics[metric_key]["independence_I3"] = res
        

    
    cluster_metrics_cols_names = ["Learnability score (PAPER version)", "Generalizability score", "LG score", "Independence score", "Compatibility", "I2", "Independence v3 (PAPER version)"]
    
    table = PrettyTable()
    table.field_names = ["Metric"] + cluster_metrics_cols_names

    for metric_key in metrics_keys:
        row = [metric_key]

        learnability_score_avg = cluster_metrics[metric_key]['learnability']['mean']
        learnability_score_std = cluster_metrics[metric_key]['learnability']['std']
        row += [f"{learnability_score_avg:0.2f}+-{learnability_score_std:0.2f}"]

        generalizability_score_avg = cluster_metrics[metric_key]['generalizability']['mean']
        generalizability_score_std = cluster_metrics[metric_key]['generalizability']['std']
        row += [f"{generalizability_score_avg:0.2f}+-{generalizability_score_std:0.2f}"]

        lg_score_avg = cluster_metrics[metric_key]['lg']['mean']
        lg_score_std = cluster_metrics[metric_key]['lg']['std']
        row += [f"{lg_score_avg:0.2f}+-{lg_score_std:0.2f}"]

        independence_score_avg = cluster_metrics[metric_key]['independence']['mean']
        independence_score_std = cluster_metrics[metric_key]['independence']['std']
        row += [f"{independence_score_avg:0.2f}+-{independence_score_std:0.2f}"]

        compatibility_score_avg = cluster_metrics[metric_key]['compatibility']['mean']
        compatibility_score_std = cluster_metrics[metric_key]['compatibility']['std']
        row += [f"{compatibility_score_avg:0.2f}+-{compatibility_score_std:0.2f}"]

        independence_score_avg = cluster_metrics[metric_key]['independence_I2']['mean']
        independence_score_std = cluster_metrics[metric_key]['independence_I2']['std']
        row += [f"{independence_score_avg:0.2f}+-{independence_score_std:0.2f}"]

        independence_score_avg = cluster_metrics[metric_key]['independence_I3']['mean']
        independence_score_std = cluster_metrics[metric_key]['independence_I3']['std']
        row += [f"{independence_score_avg:0.2f}+-{independence_score_std:0.2f}"]

        table.add_row(row)
        
    out_path = Path(cache.get_path())
    out_path.mkdir(parents=True, exist_ok=True)

    with open(str(out_path / "table.txt"), "w") as f:
        f.write(str(table))

    write_json(str(out_path / "cluster_metrics.json"), cluster_metrics)

def compute_intra_metrics(metrics, finetuning_groups, eval_groups, metric_key):

    gen_per_cluster_score = {}
    learn_per_cluster_score = {}
    lg_per_cluster_score = {}

    for finetuning_group in finetuning_groups:
        
        # Skip [all, nothing, other_epoch, ...]
        if not finetuning_group.isdigit() or int(finetuning_group) < 0:
            continue
        
        train_val = metrics["train"][finetuning_group][f"cluster_{finetuning_group}"][metric_key]
        test_val = metrics["test"][finetuning_group][f"cluster_{finetuning_group}"][metric_key]

        generalizability_score = compute_generalizability_score(train_val, test_val)
        
        gen_per_cluster_score[finetuning_group] = generalizability_score

        learn_per_cluster_score[finetuning_group] = test_val

        lg_per_cluster_score[finetuning_group] = test_val * generalizability_score


    gen_mean = np.nanmean(list(gen_per_cluster_score.values()))
    gen_std = np.nanstd(list(gen_per_cluster_score.values()))


    learn_mean = np.nanmean(list(learn_per_cluster_score.values()))
    learn_std = np.nanstd(list(learn_per_cluster_score.values()))

    lg_mean = np.nanmean(list(lg_per_cluster_score.values()))
    lg_std = np.nanstd(list(lg_per_cluster_score.values()))

    gen_metric = {
        "mean": gen_mean,
        "std": gen_std,
        "per_cluster_score": gen_per_cluster_score,

    }

    learn_metric = {
        "mean": learn_mean,
        "std": learn_std,
        "per_cluster_score": learn_per_cluster_score,

    }

    lg_metric = {
        "mean": lg_mean,
        "std": lg_std,
        "cluster_score": lg_per_cluster_score,

    }

    return gen_metric, learn_metric, lg_metric

def compute_generalizability_score(train_val, test_val):

    if np.isnan(train_val) or np.isnan(test_val):
        return np.nan
    elif train_val == 0:
        return 0 # If train is 0, it's not even a question of generalization, it doesn't learn on the train set...
    else:
        assert 0 <= train_val <= 1
        assert 0 <= test_val <= 1

        val = np.clip(train_val - test_val, 0, 1)
        val = 1 - (val/train_val)

        return val
        
def compute_inter_metric(metrics, finetuning_groups, eval_groups, metric_key):

    per_cluster_score = {}

    for finetuning_group in finetuning_groups:
        
        # Skip [all, nothing, other_epoch, ...]
        if not finetuning_group.isdigit() or int(finetuning_group) < 0:
            continue
        
        test_val_i = metrics["test"][finetuning_group][f"cluster_{finetuning_group}"][metric_key]

        test_val_others = []
        for eval_group in eval_groups:
            eval_group_id = eval_group.split("_")[1]
            
            # No cluster_outside, or cluster_-1 or cluster_{finetuning_group} (current one)
            if not eval_group_id.isdigit() or int(eval_group_id) < 0 or eval_group_id == finetuning_group:
                continue
            eval_group_val = metrics["test"][finetuning_group][eval_group][metric_key]

            test_val_others.append(eval_group_val)

        score = compute_independence_score(test_val_i, test_val_others)

        per_cluster_score[finetuning_group] = score
        
    mean = np.nanmean(list(per_cluster_score.values()))
    std = np.nanstd(list(per_cluster_score.values()))

    return {
        "mean": mean,
        "std": std,
        "per_cluster_score": per_cluster_score,
    }

def compute_independence_score(test_val_i, test_val_others):

    if np.isnan(test_val_i):
        return np.nan
    elif test_val_i == 0:
        return 0 # If current cluster is 0, other clusters are either better or same so worse score of 0
    else:
        assert 0 <= test_val_i <= 1
        vals = []

        for test_val_other in test_val_others:
            
            if np.isnan(test_val_other):
                continue # Ignore NaNs in other clusters...

            assert 0 <= test_val_other <= 1

            val_ij = np.clip(test_val_i - test_val_other, 0, 1) / test_val_i

            assert not np.isnan(val_ij) # Should never happen
            
            vals.append(val_ij)

        if len(vals) > 0:
            return np.nanmean(vals) # Also filter NaNs just in case but shouldn't happen
        else:
            return np.nan # No other usable cluster ? -> no score

# ----------------------------------------------------------------------


def compute_compatibility(metrics, finetuning_groups, eval_groups, metric_key):

    per_cluster_score = {}

    for finetuning_group in finetuning_groups:
        
        # Skip [all, nothing, other_epoch, ...]
        if not finetuning_group.isdigit() or int(finetuning_group) < 0:
            continue
        
        # train_val = metrics["train"][finetuning_group][f"cluster_outside"][metric_key]
        test_val = metrics["test"][finetuning_group][f"cluster_outside"][metric_key]

        per_cluster_score[finetuning_group] = test_val

    mean = np.nanmean(list(per_cluster_score.values()))
    std = np.nanstd(list(per_cluster_score.values()))

    return {
        "mean": mean,
        "std": std,
        "cluster_score": per_cluster_score,
    }

    
def compute_inter_metric_I2(metrics, finetuning_groups, eval_groups, metric_key):

    per_cluster_score = {}

    for finetuning_group in finetuning_groups:
        
        # Skip [all, nothing, other_epoch, ...]
        if not finetuning_group.isdigit() or int(finetuning_group) < 0:
            continue
        
        test_val_i = metrics["test"][finetuning_group][f"cluster_{finetuning_group}"][metric_key]

        test_val_others = []
        for eval_group in eval_groups:
            eval_group_id = eval_group.split("_")[1]
            
            # No cluster_outside, or cluster_-1 or cluster_{finetuning_group} (current one)
            if not eval_group_id.isdigit() or int(eval_group_id) < 0 or eval_group_id == finetuning_group:
                continue
            eval_group_val = metrics["test"][finetuning_group][eval_group][metric_key]

            test_val_others.append(eval_group_val)

        score = compute_independence_score_I2(test_val_i, test_val_others)

        per_cluster_score[finetuning_group] = score
        
    mean = np.nanmean(list(per_cluster_score.values()))
    std = np.nanstd(list(per_cluster_score.values()))

    return {
        "mean": mean,
        "std": std,
        "per_cluster_score": per_cluster_score,
    }

def compute_independence_score_I2(test_val_i, test_val_others):

    if np.isnan(test_val_i):
        return np.nan
    elif test_val_i == 0:
        return 0 # If current cluster is 0, other clusters are either better or same so worse score of 0
    else:
        assert 0 <= test_val_i <= 1
        vals = []

        for test_val_other in test_val_others:
            
            if np.isnan(test_val_other):
                continue # Ignore NaNs in other clusters...

            assert 0 <= test_val_other <= 1

            val_ij = -test_val_other

            assert not np.isnan(val_ij) # Should never happen
            
            vals.append(val_ij)

        if len(vals) > 0:
            return np.nanmean(vals) # Also filter NaNs just in case but shouldn't happen
        else:
            return np.nan # No other usable cluster ? -> no score

# $$$$$$$$$$
def compute_inter_metric_I3(metrics, finetuning_groups, eval_groups, metric_key):

    per_cluster_score = {}

    for finetuning_group in finetuning_groups:
        
        # Skip [all, nothing, other_epoch, ...]
        if not finetuning_group.isdigit() or int(finetuning_group) < 0:
            continue
        
        test_val_i = metrics["test"][finetuning_group][f"cluster_{finetuning_group}"][metric_key]

        test_val_others = []
        for eval_group in eval_groups:
            eval_group_id = eval_group.split("_")[1]
            
            # No cluster_outside, or cluster_-1 or cluster_{finetuning_group} (current one)
            if not eval_group_id.isdigit() or int(eval_group_id) < 0 or eval_group_id == finetuning_group:
                continue
            eval_group_val = metrics["test"][finetuning_group][eval_group][metric_key]

            test_val_others.append(eval_group_val)

        score = compute_independence_score_I3(test_val_i, test_val_others)

        per_cluster_score[finetuning_group] = score
        
    mean = np.nanmean(list(per_cluster_score.values()))
    std = np.nanstd(list(per_cluster_score.values()))

    return {
        "mean": mean,
        "std": std,
        "per_cluster_score": per_cluster_score,
    }

def compute_independence_score_I3(test_val_i, test_val_others):

    if np.isnan(test_val_i):
        return np.nan
    elif test_val_i == 0:
        return 0 # If current cluster is 0, other clusters are either better or same so worse score of 0
    else:
        assert 0 <= test_val_i <= 1
        vals = []

        for test_val_other in test_val_others:
            
            if np.isnan(test_val_other):
                continue # Ignore NaNs in other clusters...

            assert 0 <= test_val_other <= 1

            val_ij = test_val_i - test_val_other

            assert not np.isnan(val_ij) # Should never happen
            
            vals.append(val_ij)

        if len(vals) > 0:
            return np.nanmean(vals) # Also filter NaNs just in case but shouldn't happen
        else:
            return np.nan # No other usable cluster ? -> no score