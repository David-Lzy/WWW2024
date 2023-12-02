import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *
# from Package import HOME_LOC

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_path = os.path.join(HOME_LOC, "DATA", "UCRArchive_2018")

ADEVERSARIAL_TRAINING_path = os.path.join(HOME_LOC, "DATA", "ADEVERSARIAL")

ATTACK_OUTPUT_path = os.path.join(HOME_LOC, "OUTPUT", "attack")

TRAIN_OUTPUT_path = os.path.join(HOME_LOC, "OUTPUT", "train")

DONE_NAME = ""
DOING_NAME = ""
MODEL_NAME = "MODEL_INFO.pth"


_ = os.path.join(HOME_LOC, "CODE", "config", "default_train.json")
with open(_, "r") as file:
    DEFAULT_TRAIN_PARAMATER = json.load(file)

_ = os.path.join(HOME_LOC, "CODE", "config", "default_attack.json")
with open(_, "r") as file:
    DEFAULT_ATTACK_PARAMATER = json.load(file)

_ = os.path.join(HOME_LOC, "CODE", "config", "DEFAULT_DATA_NAME.json")
with open(_, "r") as file:
    UNIVARIATE_DATASET_NAMES = json.load(file)


# print(
#     type(DEFAULT_TRAIN_PARAMATER),
#     type(DEFAULT_ATTACK_PARAMATER),
#     type(UNIVARIATE_DATASET_NAMES),)