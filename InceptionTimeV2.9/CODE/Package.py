import logging
import os
import sys
import datetime

# Package.py: This script sets up logging, imports essential packages and custom modules for the project.

# Setting up logging
# ------------------
# Configure the logger for the application.
current_datetime = datetime.datetime.now()
formatted_log = current_datetime.strftime("%Y_%m_%d_%H_%M_%S.log")
log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LOG", formatted_log)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for logging
handler = logging.FileHandler(log_path)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Ensure the working directory is set correctly
HOME_LOC = os.path.dirname(os.path.dirname(__file__))
if not os.getcwd() == HOME_LOC:
    logging.warning("Home path not equal to work path, changing!")
    os.chdir(HOME_LOC)

# Importing necessary libraries and handling any ModuleNotFound errors
# --------------------------------------------------------------------
try:
    # List of essential Python and PyTorch libraries for machine learning
    import torch
    import torch.optim as optim
    import numpy as np
    import time
    import csv
    import json
    import pandas as pd
    import shutil
    import re
    import copy
    import inspect
    from sklearn.metrics import precision_score, recall_score, f1_score
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    from datetime import datetime
    import torch.nn as nn
    import torch.nn.functional as F
    from pprint import pprint
except ModuleNotFoundError as e:
    logging.error(str(e))
    sys.exit(-1)

# Importing custom modules for the project
# -----------------------------------------
# Warning: Do not change the order of the following packages.
try:
    sys.path.append(HOME_LOC)
    from CODE.utils.constant import *
    from CODE.utils.constant import UNIVARIATE_DATASET_NAMES as datasets
    from CODE.utils.utils import *
    from CODE.train.trainer import Trainer
    from CODE.train.classifier import *
    from CODE.train.Augmentation import Augmentation
    from CODE.attack.attacker import Attack
except ModuleNotFoundError as e:
    logging.error(e)
    sys.exit(-1)
