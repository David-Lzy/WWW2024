import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *
from Package import TRAIN_OUTPUT_path
from CODE.train.classifier import Classifier_INCEPTION


class Trainer:
    """
    The `Trainer` class facilitates the training and evaluation of an Inception-based classifier for time series data.

    This class handles the complete training process including loading data, training, evaluating, and saving the model.

    Parameters:
    - dataset (str): Name of the dataset being used for training.
    - device (torch.device): The device (e.g., 'cuda' or 'cpu') on which the model will be trained.
    - batch_size (int): Batch size for the training process.
    - epoch (int, optional): Total number of epochs for the training process. Defaults to 750.
    - loss (function, optional): Loss function used for training. Defaults to CrossEntropyLoss.
    - override (bool, optional): If True, existing training checkpoints will be overwritten.
    - defence (dict, optional): Dictionary specifying defense mechanisms. Defaults to None.
    - path_pramater (dict, optional): Dictionary to specify training path parameters.
    - continue_train (bool, optional): If True, training will continue from the last checkpoint. Defaults to False.
    - angle (bool, optional): If True, uses angle-based defense mechanism. Defaults to None.
    - Augment (dict, optional): Dictionary for data augmentation parameters. Defaults to None.
    - adeversarial_training (bool, optional): If True, uses adversarial training. Defaults to None.
    - adeversarial_path (str, optional): Path for adversarial training. Defaults to None.
    - adeversarial_resume (bool, optional): If True, resumes training from an adversarially trained model. Defaults to None.

    Attributes:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - model (nn.Module): Instance of the Inception classifier.
    - loss_function (function): Loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer for model training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.

    Methods:
    - train_and_evaluate: Main method to train and evaluate the model.
    - evaluate: Evaluates the model on the test dataset and returns metrics.
    - __train_one_epoch__: Conducts training for one epoch.

    Usage Example:
    ```
    trainer = Trainer(dataset='my_dataset', device=torch.device('cuda'), batch_size=64, epoch=100)
    trainer.train_and_evaluate()
    ```
    """
    
    def __init__(
        self,
        dataset=None,
        device=None,
        batch_size=None,
        epoch=None,
        loss=None,
        override=None,
        defence=None,
        path_pramater=None,
        continue_train=False,
        angle=None,
        Augment=None,
        adeversarial_training=None,
        adeversarial_path=None,
        adeversarial_resume=None,
        
    ):
        # adeversarial_resume means if you want to load from the trained model. if True, we load from the model trained.
        # This parameter is not about if you continue from the last epoch. It is about if you want to load from the model trained.
        defence = build_defence_dict(
            defence,
            angle,
            Augment,
            adeversarial_training)


        default_config = DEFAULT_TRAIN_PARAMATER
        self.config = copy.deepcopy(default_config)

        # Override self.config with provided parameters
        if dataset is not None:
            self.config["dataset"] = dataset
        if device is not None:
            self.config["device"] = device
        if batch_size is not None:
            self.config["batch_size"] = batch_size
        if epoch is not None:
            self.config["epoch"] = epoch
        if loss is not None:
            self.config["loss"] = loss
        if override is not None:
            self.config["override"] = override
        if defence is not None:
            self.config["defence"] = defence
        if adeversarial_training is not None:
            self.config["adeversarial_training"] = adeversarial_training
        if adeversarial_path is not None:
            self.config["adeversarial_path"] = adeversarial_path
        if adeversarial_resume is not None:
            self.config["adeversarial_resume"] = adeversarial_resume

        self.override = self.config["override"]
        self.dataset = self.config["dataset"]
        self.epoch = self.config["epoch"]
        self.batch_size = self.config["batch_size"]
        self.defence = self.config["defence"]
        self.adeversarial_training = self.config["adeversarial_training"]
        self.adeversarial_path = self.config["adeversarial_path"]
        self.adeversarial_resume = self.config["adeversarial_resume"]

        self.device = (
            self.config["device"]
            if not self.config["device"] is None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.path_parameter = (
            {key: value for key, value in self.config.items() if key in path_pramater}
            if not path_pramater is None
            else {
                key: self.config[key]
                for key in self.config["path_pramater"]
            }
        )  # 增强鲁棒性防止傻逼

        self.method_path = get_method_loc(self.path_parameter)
        self.method_path = self.method_path.replace(
            "adeversarial_training=True",
            f"adeversarial_training_from.{self.adeversarial_path}"
            ) if self.adeversarial_training else self.method_path
        self.out_dir = os.path.join(
            TRAIN_OUTPUT_path,
            self.method_path,
            self.dataset,
        )

        # ?
        _ = os.path.join(
            ADEVERSARIAL_TRAINING_path,
            self.adeversarial_path) if self.adeversarial_training else DATASET_path
        train_loader, test_loader, shape, _, nb_classes = data_loader(
            self.dataset, batch_size=self.batch_size,data_path=_
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = Classifier_INCEPTION(
            input_shape=shape,
            nb_classes=nb_classes,
            defence=self.defence,
        ).to(self.device)

        if loss in [None, "", "default", "Classifier_INCEPTION"]:
            self.loss_function = self.__CE_loss__
            self.__f__ = self.__get_f__
        elif loss in ["angle"]:
            # Prototype_target
            self.w_target = torch.full((nb_classes, nb_classes), -1 / (nb_classes - 1))
            torch.Tensor.fill_diagonal_(self.w_target, 1)
            self.w_target = self.w_target.to(self.device)

            self.loss_function = self.__angle_loss__
            self.__f__ = self.__get_f_no_w__

        else:
            raise KeyError("No right loss chosen!")

        # create_directory(self.out_dir)

        # self.loss_function = loss().to(self.device)
        self.optimizer = Adam(self.model.parameters())

        # modified
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            verbose=True,
            min_lr=0.0001,
        )

        self.resume_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        self.continue_train = continue_train

    ####################### __init_finisehd__ #######################

    def __get_f_no_w__(self, model, x_batch):
        predictions, _ = model(x_batch)
        return predictions

    def __get_f__(self, model, x_batch):
        return model(x_batch)

    def __check_resume__(self):

        def check_check_point(path):
            checkpoint = torch.load(path)
            for i_key, values in checkpoint.items():
                self.resume_dict[i_key] = values

            wanted_e = self.resume_dict["config"]["epoch"]
            real_end_e = self.resume_dict["epoch"]
            this_time_e = self.epoch
            start, end = determine_epochs(
                wanted_e,
                real_end_e,
                this_time_e,
                self.continue_train
                )
            self.epoch = end
            self.config['epoch'] = end
            return start, checkpoint
        
        def __resume__(checkpoint: dict):
            self.model.load_state_dict(
                checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"])
            logging.info(
                f"Pth file load from {checkpoint['out_dir']}")
        

        start = 1
        target_file = os.path.join(self.out_dir, MODEL_NAME)
        ADE_TRAIN_file = os.path.join(
            ADEVERSARIAL_TRAINING_path,
            self.adeversarial_path,
            self.dataset,
            MODEL_NAME)
        if os.path.exists(target_file):
            if self.override:
                logging.info(f"Del task {self.dataset} all files.")
                shutil.rmtree(self.out_dir)
            else:
                start, checkpoint = check_check_point(target_file)

                if self.adeversarial_training:
                    # This means you are try to resume from a model not used for adeversarial training.
                    if not checkpoint['adeversarial_training'] :
                        logging.warning("ADEVERSARIAL_TRAINING is not match!")
                        os.remove(target_file)
                        logging.info(f"Del task {target_file}.")
                        logging.info(f"Resume from {ADE_TRAIN_file}")
                        if self.adeversarial_resume and os.path.exists(ADE_TRAIN_file):
                            start, checkpoint = check_check_point(ADE_TRAIN_file)
                        else:
                            return start
            __resume__(checkpoint)
        create_directory(self.out_dir)
        return start

    def train_and_evaluate(self):
        start_epoch = self.__check_resume__()
        if start_epoch == -1:
            return
        test_loss_file = open(os.path.join(self.out_dir, "test_loss.txt"), "a")
        logging.info(f"Start locking File {test_loss_file.name}")

        learning_rate_file = open(os.path.join(self.out_dir, "learningRate.txt"), "a")
        logging.info(f"Start locking File {learning_rate_file.name}")

        # current_time = datetime.now()
        # logging.info(f"Current time: {current_time} \n")
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch + 1):
            self.__train_one_epoch__()
            self.__save_check_point__(
                epoch, start_time, test_loss_file, learning_rate_file
            )

            # Evaluation Phase
            test_loss = self.__cal_loss__()

            # Record test loss and learning rate
            test_loss_file.write(f"{test_loss}\n")
            learning_rate_file.write(f"{self.optimizer.param_groups[0]['lr']}\n")
            self.scheduler.step(test_loss)

        test_loss_file.close()
        learning_rate_file.close()

    def __save_check_point__(
        self, epoch, start_time, test_loss_file, learning_rate_file
    ):
        # Save model weights every 50 epochs and delete the old one
        if epoch % 50 == 0 or epoch >= self.epoch:
            checkpoint_path = os.path.join(self.out_dir, MODEL_NAME)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "config": self.config,
                    "adeversarial_training": self.adeversarial_training,
                    "out_dir": self.out_dir
                },
                checkpoint_path,
            )
            learning_rate_file.flush()
            test_loss_file.flush()
        if epoch == self.epoch:
            self.train_result["duration"] = time.time() - start_time
            self.evaluate()
            save_metrics(self.out_dir, "train", self.train_result)
            save_metrics(self.out_dir, "test", self.test_result)
            logging.info(f"Task {self.dataset} Finished")
            logging.info("-" * 80)

    def __cal_loss__(
        self,
    ):
        test_loss = 0
        for x_batch, y_batch in self.test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            loss, _ = self.loss_function(x_batch, y_batch)
            test_loss += loss.item()
        test_loss /= len(self.test_loader)
        return test_loss

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predictions = self.__f__(model, x_batch)
                loss, predictions = self.loss_function(x_batch, y_batch)
                test_loss += loss.item()
                # test_loss += loss_function(x_batch, y_batch).item()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()
                test_preds.extend(pred.squeeze().cpu().numpy())
                test_targets.extend(y_batch.cpu().numpy())

            test_loss /= len(self.test_loader)

        accuracy = correct / len(self.test_loader.dataset)
        precision, recall, f1 = metrics(test_targets, test_preds)

        self.test_result = {
            "loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def __angle_loss__(self, x_batch, y_batch):
        predictions, w_mtx = self.model(x_batch)
        # Prototype Loss
        loss_W = ((w_mtx - self.w_target) ** 2).mean()
        # CE loss
        loss_CE = CrossEntropyLoss()(predictions, y_batch)
        loss_total = loss_W + loss_CE
        return loss_total, predictions

    def __CE_loss__(self, x_batch, y_batch):
        predictions = self.model(x_batch)
        # CE loss
        loss_CE = CrossEntropyLoss()(predictions, y_batch)
        return loss_CE, predictions

    def __train_one_epoch__(self):
        self.model.train()

        train_loss = 0
        train_preds, train_targets = [], []
        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss, predictions = self.loss_function(x_batch, y_batch)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())

        train_loss /= len(self.train_loader)

        accuracy = np.mean(np.array(train_preds) == np.array(train_targets))
        precision, recall, f1 = metrics(train_targets, train_preds)

        self.train_result = {
            "loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        } 
