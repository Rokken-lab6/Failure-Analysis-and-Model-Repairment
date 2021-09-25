
from tools.utils import get_class

from model_repair.cache42.cached_42 import cache_42

from model_repair.tasks.extract_checkpoint import extract_checkpoint
from model_repair.tasks.merge_extracted_checkpoints import merge_extracted_checkpoints

from model_repair.tasks.get_true_negatives import get_true_negatives
from model_repair.override_for_release import get_interface

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def trigger_extract(cfg, gcfg, cache):

    # Load the interface (NOTE: Was overided to keep the cache valid in the publicly released version)
    interface = get_interface(gcfg)

    if cfg.epochs is None:
        checkpoints = interface.get_checkpoints()
        if cfg.stride > 1:
            checkpoints = checkpoints[::cfg.stride]
        checkpoints = checkpoints[-cfg.max_checkpoints:]
    else:
        checkpoints = []
        for epoch in cfg.epochs:
            checkpoint_path = interface.get_checkpoint_path(epoch)
            checkpoints.append(checkpoint_path)

    if cfg.epochs is None or len(cfg.epochs) > 1:
        # If 1 epoch, it can be the ref epoch..
        epochs_ids = [int(e.stem.split("model")[1]) for e in checkpoints]
        ref_epoch = int(cfg["ref_epoch"])

        if ref_epoch not in epochs_ids:
            print("adding", ref_epoch)
            checkpoint_path = interface.get_checkpoint_path(ref_epoch)
            checkpoints.append(checkpoint_path)      
    elif len(cfg.epochs) == 1:
        ref_epoch = cfg.epochs[0]
        print("ref epoch", ref_epoch)
        

    print("checkpoints:", checkpoints)
    extracteds = []
    for checkpoint in checkpoints:
        if "get_true_negatives" in gcfg:
            get_true_negatives(gcfg["get_true_negatives"], str(checkpoint), cfg, gcfg=gcfg)
            exit()

        res = extract_checkpoint(cfg["extract_checkpoint"], str(checkpoint), cfg, gcfg=gcfg)
        extracteds.append(res)

        if int(checkpoint.stem.split("model")[1]) == ref_epoch:
            ref_extracted = res

    # NOTE: This is not a cached task
    merge_extracted_checkpoints(extracteds, ref_extracted, out_path=cache.get_path())