#!/usr/bin/env python

import argparse
from pathlib import Path
import wandb
from box import Box

import tools.pudb_hook
from trainer import Trainer

def main(cfg):
    if cfg.wandb_use:
        run = wandb.init(
            project=cfg.wandb_project_name,
            sync_tensorboard=True,
            entity="PLEASE_SET",
            group=cfg.wandb_group,
            config=cfg,
            allow_val_change=True,
        )

        # Get wandb exp name
        wandb.run.save()
        wandb_name = wandb.run.name
        cfg_name = cfg.name

        run_name = f"{wandb_name}_{cfg_name}"
        cfg.name = run_name

        # Make a sub dir with the group name
        cfg.sub_dir = global_hparams.wandb_group

        # Change wandb name to include cfg name (but not split name to save space)
        run.name = f"{wandb_name}_{cfg_name}"
        wandb.run.save()
    else:
        cfg.sub_dir = global_hparams.wandb_group

    go(cfg)

def main_sweep():

    cfg = Box(dict(global_hparams))
    run = wandb.init(sync_tensorboard=True,)

    print("Sweep params", dict(wandb.config))

    # Update cfg with wandb params from the sweep
    cfg.update(dict(wandb.config))

    cfg.scheduler.use = cfg.scheduler_use_tmp
    cfg.scheduler.params.patience = cfg.scheduler_patience_tmp

    # Get wandb exp name
    wandb.run.save()
    wandb_name = wandb.run.name
    sweep_name = global_hparams.sweep_name
    split_name = cfg.split
    # Save split in the name so it's easy to get the right model during cross-val
    run_name = f"{split_name}_{wandb_name}_{sweep_name}"
    cfg.name = run_name

    # Change wandb name to include sweep name (but not split name to save space)
    run.name = f"{wandb_name}_{sweep_name}"
    wandb.run.save()

    # Make a sub dir with the sweep name
    cfg.sub_dir = global_hparams.sweep_name

    # Give back all config to wandb
    wandb.config.update(cfg)

def go(cfg):
    print(f"Run name: {cfg.name}")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', dest='model_cfg', type=str, required=True)
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--sweep", type=str, default=None)
    args = parser.parse_args()

    with open(args.model_cfg, 'r') as stream:
        global_hparams = Box.from_yaml(stream)

    if args.nowandb:
        global_hparams.wandb_use = False
    else:
        global_hparams.wandb_use = True

    if args.sweep is not None:
        global_hparams.sweep_name = args.sweep
        wandb.agent(args.sweep, function=main_sweep)
    else:
        main(global_hparams)