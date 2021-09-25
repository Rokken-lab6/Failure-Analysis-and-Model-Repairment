# -*- coding: utf8 -*-

from pathlib import Path
import numpy as np

from sumagg.sumagg import BaseSummary, SumAgg

class MeanSummary(BaseSummary):

    def __init__(self, name):
        self.name = name
        self.reset()

    def add(self, data):
        self.all_data.append(data)

    def write_metrics(self):
        sumagg = SumAgg()

        if len(self.all_data) > 0:
            log_value = np.mean(self.all_data)
        else:
            log_value = np.nan
        sumagg.tensorboard_writer.add_scalar(f"{sumagg.phase}/{self.name}", log_value, sumagg.epoch)

        sumagg.epoch_metrics[self.name] = log_value

    def reset(self):
        self.all_data = []

class LastSummary(BaseSummary):

    def __init__(self, name):
        self.name = name
        self.reset()

    def add(self, data):
        if data is not None:
            self.log_value = data

    def write_metrics(self):
        sumagg = SumAgg()

        if self.log_value is not None:
            sumagg.tensorboard_writer.add_scalar(f"{sumagg.phase}/{self.name}", self.log_value, sumagg.epoch)
            sumagg.epoch_metrics[self.name] = self.log_value

    def reset(self):
        self.log_value = None

class HistAppendSummary(BaseSummary):

    def __init__(self, name):
        self.name = name
        self.reset()

    def add(self, data):
        self.all_data.append(data)

    def write_metrics(self):
        sumagg = SumAgg()

        if len(self.all_data) > 0:
            all_data = np.array(self.all_data)
            sumagg.tensorboard_writer.add_histogram(f"{sumagg.phase}/{self.name}", all_data, sumagg.epoch)

            sumagg.epoch_metrics[self.name] = all_data

    def reset(self):
        self.all_data = []

class NoRestrictionImageSummary(BaseSummary):

    def __init__(self, name):
        self.name = name
        self.reset()

    def add(self, data):
        if data is not None: # Unless None, Overwrite
            self.log_img = data

    def write_metrics(self):
        sumagg = SumAgg()

        if self.log_img is not None:
            if len(self.log_img.shape) == 2:
                self.log_img = np.expand_dims(self.log_img, 0)

            sumagg.tensorboard_writer.add_image(f"{sumagg.phase}/{self.name}", self.log_img, sumagg.epoch)
            sumagg.epoch_metrics[self.name] = self.log_img

    def reset(self):
        self.log_img = None