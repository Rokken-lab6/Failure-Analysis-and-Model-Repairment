
import pandas as pd
from sklearn.model_selection import train_test_split

class AbstractInterface:

    def __init__(self, int_cfg):
        self.int_cfg = int_cfg

    def split_cluster(self, split_cfg, df_cluster):

        train_val, test = train_test_split(df_cluster, test_size=split_cfg["test_ratio"])
        train, val = train_test_split(train_val, test_size=split_cfg["val_to_train_ratio"])

        train = train.copy()
        val = val.copy()
        test = test.copy()
        
        train["split_id"] = 0
        val["split_id"] = 1
        test["split_id"] = 2

        df = pd.concat([train, val, test], 0).reset_index(drop=True)

        return df

    def identify_cluster(self, new_objs, old_objs):
        # cluster = -2: New object
        # otherwise same as matched object (easy for classification, less for object detection)
        raise NotImplementedError()

    def get_example_key(self):
        # Key in the sample that identify every example (an image, can contain multiple objects)
        raise NotImplementedError()

    def compute_metrics(self, objs):
        # Return a dict of metrics
        raise NotImplementedError()

    def validation_metric(self):
        # return "accuracy"
        raise NotImplementedError()