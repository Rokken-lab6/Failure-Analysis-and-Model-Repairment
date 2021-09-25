import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.utils import read_json

from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def plot_heatmaps(cfg, finetuned_metrics, gcfg, cache):

    metrics = read_json(finetuned_metrics.get_path())


    # Automatically get all the metrics that exists from the dict
    finetuning_groups = list(metrics["test"].keys())
    eval_groups = list(metrics["test"][finetuning_groups[0]].keys())
    metrics_keys = list(metrics["test"][finetuning_groups[0]][eval_groups[0]].keys())

    eval_groups = [e for e in eval_groups if e.split("_")[1][0] != "-"] # Note: Remove negative cluster

    if "sort_matrix" in cfg and cfg["sort_matrix"]:
        sort_matrix = True
    else:
        sort_matrix = False

    set_ = "train"
    set_name = "(Train set)"
    for metric_key in metrics_keys:
        human_name = get_human_name(metric_key)
        
        out_path = cache.get_path() / "only_clusters" / set_
        out_path.mkdir(parents=True, exist_ok=True)

        plot_matrix(key=metric_key, title=f"{human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix)
        plot_matrix(key=metric_key, title=f"Before finetuning: {human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, no_finetuning=True)


        out_path = cache.get_path() / "full" / set_
        out_path.mkdir(parents=True, exist_ok=True)
        plot_matrix(key=metric_key, title=f"{human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, only_clusters=False)
        plot_matrix(key=metric_key, title=f"Before finetuning: {human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, no_finetuning=True, only_clusters=False)

    set_ = "test"
    set_name = "(Test set)"
    for metric_key in metrics_keys:
        human_name = get_human_name(metric_key)
        
        out_path = cache.get_path() / "only_clusters" / set_
        out_path.mkdir(parents=True, exist_ok=True)

        plot_matrix(key=metric_key, title=f"{human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix)
        plot_matrix(key=metric_key, title=f"Before finetuning: {human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, no_finetuning=True)


        out_path = cache.get_path() / "full" / set_
        out_path.mkdir(parents=True, exist_ok=True)

        plot_matrix(key=metric_key, title=f"{human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, only_clusters=False)
        plot_matrix(key=metric_key, title=f"Before finetuning: {human_name} {set_name}", metrics=metrics[set_], out_path=out_path, finetuning_groups=finetuning_groups, eval_groups=eval_groups, sort_matrix=sort_matrix, no_finetuning=True, only_clusters=False)

def get_human_name(key):
    names = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
    }
    if key in names:
        return names[key]
    else:
        return key

def is_number(str):
    try:
        val = int(str)
    except:
        val = None

    return val is not None

def plot_matrix(key, title, metrics, out_path, finetuning_groups, eval_groups, no_scaling=False, figsize=(12, 10), no_finetuning=False, only_clusters=True, sort_matrix=False):

    if only_clusters:
        finetuning_groups = [e for e in finetuning_groups if is_number(e)]
        eval_groups = [e for e in eval_groups if is_number(e.split("_")[1]) or e.split("_")[1] == "outside"]

    N_row = len(finetuning_groups)
    N_col = len(eval_groups)


    if not only_clusters:
        M = np.ones((N_row + 1, N_col)) * -42
    else:
        M = np.ones((N_row, N_col)) * -42


    for row_i in range(N_row):
        for col_i in range(N_col):
            row_name = finetuning_groups[row_i]
            col_name = eval_groups[col_i]
            if no_finetuning:
                try:
                    group_id_int = int(row_name)
                except:
                    group_id_int = None
                if group_id_int is not None or row_name == "all":
                    row_name = "nothing"
            try:
                M[row_i][col_i] = metrics[row_name][col_name][key]
            except:
                pass

    if not only_clusters:
        for col_i in range(N_col):
            col_name = eval_groups[col_i]

            try:
                M[row_i + 1][col_i] = metrics[row_name][col_name]["count"]
            except:
                pass

    # Cosmetic eval groups
    eval_groups_names = []
    for eval_group in eval_groups:
        _, group_id = eval_group.split("_")

        try:
            group_id_int = int(group_id)
        except:
            group_id_int = None

        if group_id_int is not None:
            eval_group_name = f"Failure\n type {group_id_int + 1}"
        elif group_id == "outside":
            eval_group_name = "Non-\nfailures"
        elif group_id == "all":
            eval_group_name = "All\ndata"
        else:
            eval_group_name = "???"

        eval_groups_names.append(eval_group_name)

    # Cosmetic finetune groups
    finetuning_groups_names = []
    for finetuning_group in finetuning_groups:
        group_id = finetuning_group

        try:
            group_id_int = int(group_id)
        except:
            group_id_int = None

        if group_id_int is not None:
            finetuning_group_name = f"Finetuning\n on type {group_id_int + 1}"
        elif group_id == "all":
            finetuning_group_name = "Finetuning all"
        elif group_id == "nothing":
            finetuning_group_name = "No finetuning"
        elif group_id == "other_epoch":
            finetuning_group_name = "Other checkpoint"
        else:
            finetuning_group_name = "???"

        finetuning_groups_names.append(finetuning_group_name)

    
    if not only_clusters:
        finetuning_groups_names.append("Count")

    f, ax = plt.subplots(figsize=figsize)

    annotation_font_size = 20 if len(finetuning_groups_names) < 10 else 16

    if sort_matrix and only_clusters:
        M_to_sort = M[:, :-1]
        if np.unique(M_to_sort).shape[0] > 1:
            

            import pandas as pd
            df = pd.DataFrame(M_to_sort)

            df = df.T # SORT by row
            corrMatrix = df.corr()


            corrMatrix = corrMatrix.fillna(0).to_numpy()

            if np.unique(corrMatrix).shape[0] > 1:
                from sklearn.cluster import SpectralBiclustering
                n_clusters = M_to_sort.shape[0]
                model = SpectralBiclustering(n_clusters=n_clusters, method='bistochastic',
                                            random_state=0)
                model.fit(corrMatrix)

                idx = np.argsort(model.row_labels_)
                assert idx.shape[0] == M_to_sort.shape[0]

                fit_data = M_to_sort[idx]
                fit_data = fit_data[:, idx]

                M[:, :-1] = fit_data
                
                sorted_finetuning_groups_names = []
                sorted_eval_groups_names = []
                for k in idx:
                    sorted_finetuning_groups_names.append(finetuning_groups_names[k])
                    sorted_eval_groups_names.append(eval_groups_names[k])
                sorted_eval_groups_names.append(eval_groups_names[-1])

                finetuning_groups_names = sorted_finetuning_groups_names
                eval_groups_names = sorted_eval_groups_names

    if no_scaling:
        ax = sns.heatmap(M, annot=True, xticklabels=eval_groups_names, yticklabels=finetuning_groups_names, cmap="viridis", annot_kws={"size": annotation_font_size})
    else:
        ax = sns.heatmap(M, vmin=0, vmax=1, annot=True, xticklabels=eval_groups_names, yticklabels=finetuning_groups_names, cmap="viridis", square=True, annot_kws={"size": annotation_font_size})
    

    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 14)
    plt.title(title, fontsize=24)
    ax.set_aspect("equal")

    f.tight_layout()

    if no_finetuning:
        plt.savefig(str(out_path / f"{key}_before_finetuning.png"), dpi=400)
    else:
        plt.savefig(str(out_path / f"{key}.png"), dpi=400)
    plt.close()