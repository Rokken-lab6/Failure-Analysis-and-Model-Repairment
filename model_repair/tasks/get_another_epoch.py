import torch

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

@cache_42(ignore_args=["gcfg"], force_recompute=True)
# @cache_42(ignore_args=["gcfg"])
def get_another_epoch(cfg, split_out, gcfg, cache):
    # NOTE: split_out is there so the caching is invalidated properly

    # Load interface
    interface = get_interface(gcfg)

    # Create output path
    out_path = cache.get_path()
    out_path.mkdir(parents=True, exist_ok=True)
    filename_best_val_model = str(out_path / "model_best_val.pytorch")

    # Load model
    checkpoint_path = interface.get_checkpoint_path(cfg["other_epoch"])
    model = interface.load_model(checkpoint_path)

    # Save model
    torch.save(model.state_dict(), filename_best_val_model)

    return cache