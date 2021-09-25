

import time
import math
from pathlib import Path
import itertools
from itertools import islice
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.torch_utils
from sumagg.sumagg import SumAgg
from summaries import *
from state_management import detach_state, reset_state

class BaseTrainer:
    def __init__(self, cfg):

        self.cfg = cfg

        tools.torch_utils.ssh_mode = cfg.ssh_mode

        # Create output directory
        if cfg.only_test_mode:
            model_out_path = Path(cfg.test_mode_model_out_path)
            summary_out_path = Path(cfg.test_mode_out_path)
        elif cfg.sub_dir is None:
            model_out_path = Path(cfg.out_path) / cfg.name
            summary_out_path = model_out_path
        else:
            model_out_path = Path(cfg.out_path) / cfg.sub_dir / cfg.name
            summary_out_path = model_out_path

        cfg.model_out_path = str(model_out_path)

        print(f"Saving to {model_out_path}.")
        model_out_path.mkdir(parents=True, exist_ok=True)
        summary_out_path.mkdir(parents=True, exist_ok=True)

        # Save config file in output directory
        if not cfg.only_test_mode:
            cfg.to_yaml(str(model_out_path / "cfg.yml"))
        self.model_out_path = model_out_path

        # Create instance of summary writer
        SumAgg(cfg, summary_out_path)
        SumAgg().add_summaries([
            MeanSummary("loss"),
            HistAppendSummary("grad_norm"),
            LastSummary("inference_time"),
            LastSummary("fps"),
        ])
        self.declare_summaries() # Specific to model

        # Load model
        self.model = self.load_model()
        if self.cfg.use_swa and not cfg.only_test_mode:
            import torch.optim.swa_utils 
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)

        # Create optimizer
        if not cfg.only_test_mode:
            self.model_parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = self.create_optimizer(self.model_parameters)
            self.scheduler = self.create_scheduler(self.optimizer)

        # Dataloaders
        self.dataloader_train, self.dataloader_val, self.dataloader_test = self.get_dataloaders()

        self.best_score = -10**6
        self.epoch = 0


    # Public: --------------------------------------
    def train(self):
        # Launch training loop
        self._training_loop()

        # NOTE: _do_test() will assume there won't be anymore training and will overwrite the model
        self._do_test()

    # Internal: ----------------------------------------
    def _training_loop(self):

        self.start_time = time.time()
        self.last_best_score_time = self.start_time

        for epoch, phase in itertools.product(range(self.cfg.max_epochs), ["Train", "Val"]):
            self.epoch = epoch

            if self.cfg.test_every_epoch and phase == "Val":
                self._do_test(use_best_val=False)

            self.phase = phase
            self.training = (phase == "Train")
            self.dataloader = (
                self.dataloader_train if self.training else self.dataloader_val
            )

            # Stop conditions
            if self._check_max_time_elapsed() or self._check_is_not_learning() or self._check_is_not_improving():
                break

            self._do_epoch()


    def _do_epoch(self):
        print("\nEpoch: {}, Phase: {}".format(self.epoch, self.phase))

        SumAgg().start_epoch(self.phase, self.epoch)

        self.model.train(self.training)
        torch.set_grad_enabled(self.training)

        len_dataloader = len(self.dataloader)

        if self.cfg.max_batch_epoch is not None:
            dataloader = islice(self.dataloader, self.cfg.max_batch_epoch)
            n_batches = self.cfg.max_batch_epoch
        else:
            n_batches = len_dataloader

        summary_iters = (
            np.linspace(
                0,
                n_batches,
                num=int(self.cfg.summary_proportion * n_batches),
                endpoint=False,
            )
            .astype(int)
            .tolist()
        )

        for i_batch, sample in tqdm(enumerate(dataloader), total=len_dataloader):
            self.compute_summary = i_batch in summary_iters
            SumAgg().set_ignore_iter(not self.compute_summary)

            # Reset state
            state = {}
            state = reset_state(state)

            f_sample = self._format_sample(sample)
            sample_chunks = self._chunk_sample(f_sample, self.cfg.chunk_size)

            for i_chunk, sample_chunk in enumerate(sample_chunks):
                sample_chunk = self._prepare_sample(sample_chunk)
                state = self._do_training_iter(sample_chunk, state)
                
                # Detach state
                if (i_batch + 1) % self.cfg.detach_modulo == 0:
                    state = detach_state(state)

        # End of epoch code
        if not self.training and self.epoch == 0:
            sample_chunks = self._chunk_sample(f_sample, 1) # Chunk of size 1 to check the speed !
            sample_chunk = self._prepare_sample(sample_chunks[0])
            self._do_speed_test(sample_chunk, state)

        SumAgg().end_epoch()

        if self.training and self.cfg.use_swa and self.epoch >= self.cfg.swa_start: 
            self.swa_model.update_parameters(self.model)

        if self.epoch % (self.cfg.save_model_step) == 0 and not self.training:
            filename = str(self.model_out_path / f"model{self.epoch}.pytorch")
            torch.save(self.model.state_dict(), filename)
        
        if self.cfg.reset_weights_modulo is not None and self.epoch % self.cfg.reset_weights_modulo == 0:
            self.model.reset_weights()
        
        if not self.training:
            
            current_score = self.get_current_score() # Must be called after end_epoch()

            if self.scheduler is not None:
                if self.cfg.scheduler.give_val:
                    self.scheduler.step(current_score)
                else:
                    self.scheduler.step()

            if current_score > self.best_score or math.isnan(self.best_score) or self.epoch == 0:
                self.best_score = current_score
                self.last_best_score_time = time.time()

                print(f"New best score: {self.best_score} at epoch {self.epoch} !")

                if self.cfg.use_swa:
                    state_dict = self.swa_model.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # Save best model (even if not epoch that should be saved)
                filename = str(self.model_out_path / "best_model.pytorch")
                torch.save(state_dict, filename)

                # Save to old Pytorch format just in case there is a problem for the test
                filename = str(self.model_out_path / "best_model_old_pytorch.pytorch")
                torch.save(state_dict, filename, _use_new_zipfile_serialization=False)

    def _do_training_iter(self, sample, state):

        y = self.forward(sample, state)
        state = y.state
        loss = self.compute_loss(sample, y)

        SumAgg().add("loss", loss.item())

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()

            max_grad_norm = self.cfg.max_grad_norm
            if max_grad_norm is not None:
                params = self.model.parameters()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                SumAgg().add("grad_norm", grad_norm.item())

            self.optimizer.step()

        return state
        

    def _do_test(self, use_best_val=True):

        print("\nTesting...")

        self.phase = "Test"
        self.training = False
        self.dataloader = self.dataloader_test
        torch.set_grad_enabled(self.training)
        
        if use_best_val:
            # Overwrite training model and load best model weights
            self.model = self.load_model()

            filename = str(self.model_out_path / "best_model.pytorch")
            
            if self.cfg.use_swa: # Best model on validation without SWA though
                print("Using SWA model for eval")
                self.swa_model.load_state_dict(torch.load(filename))
                self.model = self.swa_model
            else:
                self.model.load_state_dict(torch.load(filename))

        self.model.eval()

        SumAgg().start_epoch(self.phase, self.epoch)

        self.model.train(self.training)

        if self.cfg.smaller_test is not None:
            print(f"WARNING: Using smaller_test set representing {self.cfg.smaller_test/len(self.dataloader)}%")
            test_set_len = self.cfg.smaller_test
        else:
            test_set_len = len(self.dataloader)

        state = {}
        for i_batch, sample in tqdm(enumerate(self.dataloader), total=test_set_len):

            if self.cfg.smaller_test is not None and i_batch > test_set_len:
                break

            f_sample = self._format_sample(sample)
            sample_chunks = self._chunk_sample(f_sample, self.cfg.chunk_size)

            state = reset_state(state)
            for sample_chunk in sample_chunks:
                sample_chunk = self._prepare_sample(sample_chunk)

                y = self.forward(sample_chunk, state)
                state = y.state

                loss = self.compute_loss(sample_chunk, y)

                SumAgg().add("loss", loss.item())

            if self.cfg.quick_debug and i_batch > 3:
                print("Skipping full test infering because of quick_debug mode...")
                break

        SumAgg().end_epoch()

    def _do_speed_test(self, p_sample, state):
        print("Doing speed test...")
        # Warm-up:
        y = self.forward(p_sample, state)
        y = self.forward(p_sample, state)
        y = self.forward(p_sample, state)

        # Speed test:
        start_time = time.time()
        for i in range(self.cfg.speed_test_iters):
            y = self.forward(p_sample, state)

        duration_ms = ((time.time() - start_time) * 1000) / self.cfg.speed_test_iters
        SumAgg().add("inference_time", duration_ms)
        SumAgg().add("fps", 1000 / duration_ms)

        print(f"Speed test: {duration_ms} ms")
        print(f"Speed test: {1000 / duration_ms} FPS")


    def _check_max_time_elapsed(self):

        # Maximum time stop condition
        elapsed = False
        if self.training and self.cfg.max_time is not None:
            total_time = time.time() - self.start_time
            # max_time is in hours
            max_time_seconds = self.cfg.max_time * 60 * 60
            if total_time > max_time_seconds:
                try:
                    total_time_str = str(timedelta(seconds=round(total_time)))
                except:
                    total_time_str = "ERROR str" # TODO: check ok
                print(f"{total_time_str} elapsed, stopping training...")
                elapsed = True

        return elapsed

    
    def _check_is_not_learning(self):
        # Check if the model couldn't learn since the beginning

        is_not_learning = False

        # Smart give-up stop condition
        if not self.training and self.cfg.time_smart_give_up is not None:

            total_time = time.time() - self.start_time
            # time is in hours
            time_smart_give_up_seconds = self.cfg.time_smart_give_up * 60 * 60
            if total_time > time_smart_give_up_seconds:
                if self.best_score < self.cfg.min_score_not_give_up:
                    total_time_str = str(
                        timedelta(seconds=round(total_time))
                    )
                    print(
                        f"Best score: {self.best_score} after {total_time_str}, giving-up..."
                    )
                    is_not_learning = True
                    
        return is_not_learning


    def _check_is_not_improving(self):
        # Check if the model isn't improving anymore

        is_not_improving = False
        
        if not self.training and self.cfg.time_since_last_improvement is not None:
          
            total_time = time.time() - self.last_best_score_time
            print(f"DEBUG: Total time since last best score: {total_time/(60*60)} hours.")

            # time is in hours
            time_since_last_improvement_seconds = self.cfg.time_since_last_improvement * 60 * 60
            
            if total_time > time_since_last_improvement_seconds:
                total_time_str = str(
                    timedelta(seconds=round(total_time))
                )
                print(
                    f"Didn't improve after {total_time_str}, giving-up..."
                )
                is_not_improving = True
                    
        return is_not_improving



    # To implement: ----------------------------------------------------------

    def declare_summaries(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def create_optimizer(self, model_parameters):
        raise NotImplementedError()
    def create_scheduler(self, optimizer):
        raise NotImplementedError()

    def get_dataloaders(self):
        raise NotImplementedError()

    def _format_sample(self, sample):
        raise NotImplementedError()

    def _chunk_sample(self, sample, chunk_size):
        raise NotImplementedError()

    def _prepare_sample(self, sample):
        raise NotImplementedError()

    def forward(self, sample):
        raise NotImplementedError()

    def compute_loss(self, sample, y):
        raise NotImplementedError()

    def get_current_score(self):
        raise NotImplementedError()
