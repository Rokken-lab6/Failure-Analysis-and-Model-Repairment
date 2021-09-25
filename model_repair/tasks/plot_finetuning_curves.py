import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.utils import read_json

from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def plot_finetuning_curves(cfg, finetuned_metrics, gcfg, cache):
    
    out_path = cache.get_path()

    try:
        finetuned_metrics = read_json(finetuned_metrics.get_path())
        finetuneds = finetuned_metrics["stats_finetuning"].keys()
    except:
        print("No stats_finetuning info...")
        return

    for set_ in ["val", "train"]:
        for finetuned in finetuneds:
            stats = finetuned_metrics["stats_finetuning"][finetuned]
            if stats is None:
                continue

            n_epochs = max([int(e) for e in stats[set_].keys()])

            metrics = []
            if stats[set_]["0"]["all"] is not None:
                metrics += list(stats[set_]["0"]["all"].keys())
            if stats[set_]["0"]["correct"] is not None:
                metrics += list(stats[set_]["0"]["correct"].keys())
            if stats[set_]["0"]["cluster"] is not None:
                metrics += list(stats[set_]["0"]["cluster"].keys())
            metrics = list(set(metrics))
                

            for metric in metrics:

                data_all = [None for e in range(n_epochs)]
                data_cluster = [None for e in range(n_epochs)]
                data_correct = [None for e in range(n_epochs)]
                for epoch in range(n_epochs):
                    stats[set_][str(epoch)]

                    try:
                        data_all[epoch] = stats[set_][str(epoch)]["all"][metric]
                    except:
                        pass

                    try:
                        data_cluster[epoch] = stats[set_][str(epoch)]["cluster"][metric]
                    except:
                        pass

                    try:
                        data_correct[epoch] = stats[set_][str(epoch)]["correct"][metric]
                    except:
                        pass
                
                plt.plot([e for e in range(n_epochs)], data_all, label="all")
                plt.plot([e for e in range(n_epochs)], data_cluster, label="cluster")
                plt.plot([e for e in range(n_epochs)], data_correct, label="correct")
                plt.legend(loc="lower right")
                plt.title(f"Repair on {finetuned}: {metric} {set_} set")

                path_fig = out_path / f"{finetuned}" / f"{set_}_{metric}.png"
                path_fig.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(path_fig))
                plt.close()