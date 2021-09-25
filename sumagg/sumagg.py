# -*- coding: utf8 -*-

from torch.utils.tensorboard import SummaryWriter

from .singleton import singleton

class BaseSummary:

    def add(self, data):
        # Called many times per epoch, must save data
        raise NotImplementedError()

    def write_metrics(self):
        # Perform final computation and write the summary
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

@singleton
class SumAgg:
    def __init__(self, cfg=None, out_path=None, disable=False):

        if disable:
            self.disabled = True
            self.ignore_iter = True
            return
        else:
            self.disabled = False

        # First call to SumAgg must give cfg and out_path
        assert cfg is not None
        assert out_path is not None
        
        self.cfg = cfg
        self.out_path = out_path

        self.tensorboard_writer = SummaryWriter(log_dir=str(out_path / "tensorboard"))

        self.summaries = {}

        self.epoch_metrics = {}

    def add_summary(self, summary):
        if self.disabled:
            return

        self.summaries[summary.name] = summary

    def add_summaries(self, summaries):
        if self.disabled:
            return

        for s in summaries:
            self.add_summary(s)

    def start_epoch(self, phase, epoch):
        if self.disabled:
            return

        self.phase = phase
        self.epoch = epoch
        self.ignore_iter = False

        # Reset data
        for s_name in self.summaries:
            s = self.summaries[s_name]
            s.reset()

    def set_ignore_iter(self, ignore_iter):
        if self.disabled:
            return

        self.ignore_iter = ignore_iter

    def is_compute_summary(self):
        return not self.ignore_iter

    def add(self, name, data, force=False):
        if self.disabled:
            return

        s = self.summaries[name]
        if not self.ignore_iter or force:
            if s.add is not None:
                s.add(data)

    def update(self, data_dict, force=False):
        if self.disabled:
            return

        for name in data_dict:
            self.add(name, data_dict[name], force)

    def end_epoch(self):
        if self.disabled:
            return

        for s_name in self.summaries:
            s = self.summaries[s_name]
            s.write_metrics()

    def close(self):
        if self.disabled:
            return

        self.tensorboard_writer.close()
