import sys
import os
import numpy as np
import torch
import pandas as pd
import os
import csv
import logging
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *
from Package import HOME_LOC
from CODE.utils.constant import UNIVARIATE_DATASET_NAMES as datasets
from CODE.utils.constant import DATASET_path
from CODE.utils.constant import TRAIN_OUTPUT_path
from CODE.train.Augmentation import Augmentation


def readucr(filename, delimiter="\t"):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def map_label(y_data):
    unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
    mapped_labels = np.arange(len(unique_classes))[inverse_indices]
    return mapped_labels


def load_data(
        dataset, 
        phase="TRAIN", 
        batch_size=128, 
        data_path=DATASET_path
        ):
    x, y = readucr(os.path.join(data_path, dataset, f"{dataset}_{phase}.tsv"))
    y = map_label(y)
    nb_classes = len(set(y))
    shape = x.shape
    x = x.reshape(shape[0], 1, shape[1])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        x_tensor, torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(phase == "TRAIN")
    )

    return loader, x_tensor.shape, nb_classes


def data_loader(dataset, batch_size=128, data_path=DATASET_path):
    train_loader, train_shape, nb_classes = load_data(
        dataset,  "TRAIN", batch_size=batch_size, data_path = data_path
    )
    test_loader, test_shape, nb_classes = load_data(
        dataset, "TEST", batch_size=batch_size, data_path = data_path
    )

    return train_loader, test_loader, train_shape, test_shape, nb_classes


def metrics(targets, preds):
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return precision, recall, f1


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        logging.info(f"Path {directory_name}' Created")
    else:
        logging.info(f"Path {directory_name}' Existed")


def save_metrics(directory_name, phase, metrics):
    with open(f"{directory_name}/{phase}_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

# This function is used to summary all the results of all dataset.
def concat_metrics_train(mode="train", method="", datasets=datasets):
    metrics_dfs = []
    for dataset in datasets:
        file_path = os.path.join(
            TRAIN_OUTPUT_path, method, dataset, f"{mode}_metrics.csv"
        )

        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)
        else:
            logging.warning(f"'{file_path}' not found! Skip.")
            return

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    final_df.to_csv(os.path.join(
        TRAIN_OUTPUT_path,
        f"{mode.upper()}_{'_'.join(method.split(os.path.sep))}_metrics.csv"), index=False)

# This function is used to summary all the attack results of all dataset.
def concat_metrics_attack(method, datasets=datasets):
    metrics_dfs = []
    for dataset in datasets:
        file_path = os.path.join(ATTACK_OUTPUT_path, method, dataset, "results.csv")
        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)
        else:
            logging.error(f"{file_path} has no data, ignored!")
            return

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    _ = os.path.join(
        ATTACK_OUTPUT_path, f"{'_'.join(method.split(os.path.sep))}_metrics.csv"
    )
    final_df.to_csv(_, index=False)


def write_attack_metrics_to_csv(csv_path, *args):
    metrics = {
        "ASR": args[0],
        "Mean Success Distance": args[1],
        "Mean Fail Distance": args[2],
        "Mean All Distance": args[3],
        "Success Count": args[4],
        "Fail Count": args[5],
        "Duration": args[6],
    }

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics.keys())  # Write the metric names (column names)
        writer.writerow(metrics.values())  # Write the corresponding metric values


def get_method_loc(methods):
    if type(methods) == dict:
        _new_methos = dict()
        for key, values in methods.items():
            if check_loop(values):
                if len(values) == 0:
                    pass
                else:
                    _new_methos[key] = values
            _new_methos[key] = values
    elif type(methods) in [list, tuple]:
        _new_methos = []
        for values in methods:
            if check_loop(values):
                if len(values) == 0:
                    pass
                else:
                    _new_methos.append(values)
            _new_methos.append(values)
    else:
        _new_methos = methods
    methods = _new_methos
    s_clean = re.sub(r"[{}'\", \[\]]+", "_", str(methods))
    s_clean = re.sub(r"_:_", "=", s_clean)
    s_clean = re.sub(r"__+", "_", s_clean)
    s_clean = s_clean.strip("_")
    return s_clean

def check_loop(_):
    try:
        _.__iter__
        if type(_) == str:
            return 0
    except AttributeError:
        return 0
    else:
        return 1

def build_defence_dict(
    defence,
    angle,
    Augment,
    adeversarial_training,
    ):
    def check_sub(_):
        new_Aug = dict()

        if type(_) == dict:
                new_Aug = _
        elif type(_) in (tuple, list):
            if len(_) == 2:
                if [check_loop(_[0]),
                    check_loop(_[1])
                    ] == [1, 1]:
                    new_Aug = {
                        "ahead_model": _[0],
                        "in_model": _[1],
                    }
                elif [check_loop(_[0]),
                    check_loop(_[1])
                    ] == [0, 0]:
                        new_Aug = {"ahead_model": _,}
            elif not bool(sum([check_loop(i) for i in _])):
                new_Aug = {"ahead_model": _,}
            new2_Aug = dict()
            for i, values in new_Aug.items():
                new2_Aug[i] = dict()
                for index, j in enumerate(values) :
                    name = Augmentation.get_index()[j]
                    new2_Aug[i][f'{index}.{name}'] = dict()
                    # If you want to change something here, 
                    # remember go to Augmentation.py 
                    # and  classifier.py __AUG_1__ and __AUG_2__
                    # to change the corresponding code.

            for i in list(new2_Aug.keys()):
                try:
                    if len(new2_Aug[i]) == 0:
                        del new2_Aug[i]
                except TypeError:
                    pass
            new_Aug = new2_Aug
        else:
            raise ValueError("Augmentation is not valid!")
        return new_Aug

    if type(defence) == dict:
        if defence.get("Augmentation", False):
            defence['Augmentation'] = check_sub(defence['Augmentation'])
    else:
        defence = {
            'Augmentation': dict()
        }
        if not angle == None:
            defence['angle'] = angle
        if not adeversarial_training == None:
            defence['adeversarial_training'] = adeversarial_training
        if not Augment == None:
            defence['Augmentation'] = check_sub(Augment)
        if len(defence['Augmentation']) == 0:
            del defence['Augmentation']
    # print(defence)
    return defence if len(defence) > 0 else "None"


def determine_epochs(wanted_e, real_end_e, this_time_e, continue_train):
    # If the actual number of epochs run exceeds the expected number of epochs
    if real_end_e > wanted_e:
        logging.error(f"Epochs not match! {real_end_e} > {wanted_e}. This should never happen!")
        raise ValueError(f"Epochs not match! {real_end_e} > {wanted_e}. This should never happen!")

    if continue_train:
        logging.info(f"Continuing training from epoch {real_end_e} with an additional {this_time_e} epochs.")
        return real_end_e + 1, wanted_e + this_time_e

    # If the number of epochs this time is less than or equal to the actual number of epochs
    if this_time_e < real_end_e:
        logging.error(f"You cannot train fewer epochs than the last time! {wanted_e}, {real_end_e} > {this_time_e}")
        raise ValueError(f"You cannot train fewer epochs than the last time! {wanted_e}, {real_end_e} > {this_time_e}")

    # If all epoch numbers match and are equal
    if real_end_e == wanted_e == this_time_e:
        logging.info(f"Task already done. All epochs match: {real_end_e}.")
        return -1, -1

    # If the actual and expected epoch numbers match, but this time the epoch number exceeds the expected epoch number
    if real_end_e == wanted_e:
        logging.warning(f"Epoch mismatch! Expected {wanted_e}, but got {this_time_e}. Overriding to {this_time_e}.")
        return real_end_e + 1, this_time_e

    # If the actual number of epochs is less than the expected number of epochs
    if real_end_e < wanted_e:
        logging.warning(f"The model was not trained to the end in the previous run. Expected {wanted_e}, but got {real_end_e}.")
        logging.info(f"Resuming training from epoch {real_end_e + 1} for {this_time_e} epochs.")
        return real_end_e + 1, this_time_e

    logging.error(f"Unexpected condition encountered.")
    return -1, -1


if __name__ == "__main__":
    concat_metrics_attack()
