import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Package import *
from CODE.train.classifier import Classifier_INCEPTION


class Attack(torch.nn.Module):
    """
    The `Attack` class implements adversarial attacks against an Inception-based classifier
    for time series data. It serves as a base class for specific attack implementations.

    Initialization:
    attack_instance = Attack(dataset, Model=Classifier_INCEPTION, batch_size=64, epoch=1000, eps_init=0.001, eps=0.1, device=None)

    Parameters:
    - dataset (str): Name of the dataset being used.
    - Model (torch.nn.Module, optional): Classifier model to be attacked. Defaults to Classifier_INCEPTION.
    - batch_size (int, optional): Batch size for processing. Defaults to 64.
    - epoch (int, optional): Total number of epochs for the attack. Defaults to 1000.
    - eps_init (float, optional): Initial perturbation magnitude. Defaults to 0.001.
    - eps (float, optional): Maximum allowed perturbation. Defaults to 0.1.
    - device (torch.device, optional): Device to perform the attack on (e.g., 'cuda' or 'cpu'). Defaults to None.

    Attributes:
    - model_weight_path (str): Path to the trained model weights.
    - loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - model (Classifier_INCEPTION): Instance of the Inception classifier.
    - model_name (str): Name of the model file.
    - out_dir (str): Directory to save the attack results.
    - dataset (str): Name of the dataset.
    - epoch (int): Total number of epochs for the attack.
    - eps_init (float): Initial perturbation value.
    - eps (float): Maximum allowed perturbation.
    - device (torch.device): Device on which the attack will be performed.

    Methods:
    - f(x): Returns the model's output for input x.
    - _loss_function(x, r, y_target): Abstract method defining the loss function. Must be overridden in subclasses.
    - _get_y_target(x): Abstract method for getting target labels. Must be overridden in subclasses.
    - folder_contains_files(folder_path, *file_names): Checks if a folder contains specific files.
    - __init_r__(x): Initializes the perturbation tensor r for input x.
    - __get_optimizer__(r): Returns the optimizer for the perturbation tensor r.
    - __perturb__(x): Performs the adversarial attack on input x.
    - perturb(): Performs the adversarial attack on the entire dataset.
    - metrics(): Calculates various metrics for the attack.
    - save(): Saves the attack results.
    - perturb_all(to_device=True, override=False, test=False): Performs the attack on the entire dataset and saves results.
    - build_adeversarial_training_data(): Prepares data for adversarial training.

    Notes:
    - Dependencies like Package and Classifier_INCEPTION must be accessible in the directory structure.
    - This class is intended as a base class; specific attack methods should be implemented in subclasses.
    - It provides methods for initializing perturbations, performing attacks, computing metrics, and saving results.
    """

    def __init__(
        self,
        dataset=None,
        Model=None,
        batch_size=None,
        epoch=None,
        eps_init=None,
        eps=None,
        device=None,
        train_method_path=None,  # know train_method pth location
        path_parameter=None,  # know attack output location
        adeversarial_training = None,
    ):
        super(Attack, self).__init__()

        default_config = DEFAULT_ATTACK_PARAMATER
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
        if eps_init is not None:
            self.config["eps_init"] = eps_init
        if eps is not None:
            self.config["eps"] = eps
        if adeversarial_training is not None:
            self.config["adeversarial_training"] = adeversarial_training
            logging.warning(f"adeversarial_training: {adeversarial_training}") if adeversarial_training else None
        if train_method_path is not None:
            self.config["train_method_path"] = train_method_path
        if path_parameter is not None:
            self.config["path_parameter"] = {
                key: value for key, value in self.config.items() if key in path_parameter}
        else:
            self.config["path_parameter"] = {
                key: value for key, value in self.config.items() if key in self.config["path_parameter"]}


        self.config["Model"] = Classifier_INCEPTION if Model == "Classifier_INCEPTION" else Classifier_INCEPTION if self.config["Model"] == "Classifier_INCEPTION" else Model
        # Remember change the type from str to class

        self.model = self.config["Model"]
        self.dataset = self.config["dataset"]
        self.epoch = self.config["epoch"]
        self.eps_init = self.config["eps_init"]
        self.eps = self.config["eps"]
        self.batch_size = self.config["batch_size"]
        self.path_parameter = self.config["path_parameter"]
        self.adeversarial_training = self.config["adeversarial_training"]
        self.train_method_path = self.config["train_method_path"]

        self.device = (
            self.config["device"]
            if self.config["device"] != None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.model_weight_path = os.path.join(
            TRAIN_OUTPUT_path,
            train_method_path,
            self.dataset,
            MODEL_NAME
        ) #if not train_method_path is None else train_method_path

        self.attack_method_path = get_method_loc(self.path_parameter)
        self.out_dir = os.path.join(
            ATTACK_OUTPUT_path,
            self.train_method_path,
            self.attack_method_path,
            self.dataset,
        )# We calso need train_method_path to know who we are attck.


        try:
            self.model_info = torch.load(self.model_weight_path)

            epoch = self.model_info["epoch"]

            train_config = self.model_info["config"]
            self.defence = train_config["defence"]

            try:
                if not train_config['epoch'] == epoch:
                    logging.warning(f"Epoch is not equal to the model_info['epoch'] in {self.model_weight_path}")
                if not self.train_method_path == train_config['method_path']:
                    logging.info(f"train_method_path is not equal to the model_info['method_path'] in {self.model_weight_path}")
            except KeyError:
                pass
        except FileNotFoundError:
            # Compatible with older versions
            logging.warning(f"Can't find {self.model_weight_path}, try to find {MODEL_NAME} use old method.")
            self.model_weight_path = os.path.join(
                TRAIN_OUTPUT_path,
                train_method_path,
                self.dataset,
                'Done',
                'final_model_weights.pth'
            )
            self.model_info = dict()
            self.model_info['model_state_dict'] = torch.load(self.model_weight_path)
            self.defence = None

        _phase = "TRAIN" if self.adeversarial_training else "TEST"
        self.loader, self.shape, self.nb_classes = load_data(
            self.dataset, phase=_phase, batch_size=self.batch_size
        )
        self.model = self.config["Model"](
            input_shape=self.shape,
            nb_classes=self.nb_classes,
            defence=self.defence
        )
        # Be careful of the order. load_state_dict must after model = Model()
        self.model.load_state_dict(self.model_info['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.model_name = os.path.basename(__file__).split(".")[0]

    def f(self, x):
        return self.model(x)

    def _loss_function(self, x, r, y_target):
        raise TypeError

    def _get_y_target(self, x):
        raise TypeError

    def folder_contains_files(self, folder_path, *file_names):
        try:
            folder_files = os.listdir(folder_path)
        except FileNotFoundError:
            return False
        file_names_set = set(file_names)
        ## Traverse the list of file names 
        # and check if they all exist in the folder
        for file_name in file_names_set:
            if file_name not in folder_files:
                return False
        return True

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

    def __perturb__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target = self._get_y_target(x).to(self.device)
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            loss = self._loss_function(x, r, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.info(f"Epoch: {epoch+1}/{self.epoch}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def perturb(self):
        logging.info("_" * 50)
        logging.info(f"Doing: {self.dataset}")
        start = time.time()
        all_perturbed_x = []
        all_perturbed_y = []
        all_predicted_y = []
        self.all_sum_losses = np.zeros(self.epoch)
        self.dist = []

        i = 1
        for x, y in self.loader:
            logging.info(f"batch: {i}")
            logging.info(">" * 50)
            perturbed_x, perturbed_y, predicted_y, sum_losses = self.__perturb__(x)
            perturbed_x = perturbed_x.detach().cpu().numpy()
            perturbed_x = np.squeeze(perturbed_x, axis=1)
            self.dist.append(
                np.sum((perturbed_x - np.squeeze(x.numpy(), axis=1)) ** 2, axis=1)
            )
            all_perturbed_x.append(perturbed_x)
            perturbed_y = perturbed_y.detach().cpu().numpy()
            all_perturbed_y.append(perturbed_y)
            predicted_y = predicted_y.detach().cpu().numpy()
            all_predicted_y.append(predicted_y)

            self.all_sum_losses += sum_losses
            i += 1

        self.duration = time.time() - start
        self.x_perturb = np.vstack(all_perturbed_x)
        self.y_perturb = np.hstack(all_perturbed_y)
        self.y_predict = np.vstack(all_predicted_y).argmax(axis=1)

    def metrics(self):
        map_ = self.y_perturb != self.y_predict
        self.nb_samples = self.x_perturb.shape[0]

        Count_Success = sum(map_)
        Count_Fail = self.nb_samples - Count_Success
        ASR = Count_Success / self.nb_samples
        distance = np.hstack(self.dist)
        success_distances = distance[map_]
        failure_distances = distance[~map_]

        # Create a DataFrame with the data

        self.data = {
            "ASR": ASR,
            "mean_success_distance": np.mean(success_distances),
            "mean_failure_distance": np.mean(failure_distances),
            "overall_mean_distance": np.mean(distance),
            "Count_Success": Count_Success,
            "Count_Fail": Count_Fail,
            "duration": self.duration,
        }

    def save(self):
        # Save as CSV file
        with open(
            os.path.join(self.out_dir, "results.csv"), mode="w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(self.data.keys())  
            writer.writerow(self.data.values()) 

        with open(
            os.path.join(self.out_dir, "x_perturb.tsv"), "w", newline=""
        ) as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")
            for row in self.x_perturb:
                writer.writerow(row)

        with open(os.path.join(self.out_dir, "y_perturb.npy"), "wb") as f:
            np.save(f, self.y_perturb)

        # np.save(os.path.join(self.out_dir, "y_perturb.npy"), self.y_perturb)

        with open(os.path.join(self.out_dir, "loss.txt"), "w") as f:
            _ = self.all_sum_losses / self.nb_samples
            all_mean_losses = _.reshape(-1, 1)
            np.savetxt(f, all_mean_losses, delimiter="\t")

        # np.savetxt(
        #     os.path.join(self.out_dir, "loss.txt"), all_mean_losses, delimiter="\t"
        # )
        logging.info(f"Done: {self.dataset}")
        logging.info(">" * 50)

    def perturb_all(self, override=False, to_device=False):
        _ = self.folder_contains_files(
            self.out_dir,
            "results.csv",
            "x_perturb.tsv",
            "y_perturb.npy",
            "loss.txt",
        )
        if to_device and (not override) and _:
            logging.info(f"Dataset: {self.dataset} exist, skip!")
            return
        self.perturb()
        self.metrics()
        if to_device:
            create_directory(self.out_dir)
            self.save()

    def build_adeversarial_training_data(self,):

        _ = os.path.join(
        ADEVERSARIAL_TRAINING_path,
        self.attack_method_path,
        self.dataset,
        )
        create_directory(_)

        shutil.copy(
            os.path.join(
                DATASET_path,
                self.dataset,
                f"{self.dataset}_TEST.tsv"),
            os.path.join(
                _, f"{self.dataset}_TEST.tsv"))

        shutil.copy(
            self.model_weight_path,
            os.path.join(
                _,
                MODEL_NAME)
        )

        checkpoint_path = os.path.join(_, MODEL_NAME)
        checkpoint = torch.load(checkpoint_path)
        checkpoint["attack_method"] = self.attack_method_path
        torch.save(
            checkpoint,
            checkpoint_path,
        )

        # Read two tsv files
        df1 = pd.read_csv(
            os.path.join(
                DATASET_path,
                self.dataset,
                f"{self.dataset}_TRAIN.tsv"),
            sep="\t", header=None
            )

        try:
            _df1 = pd.DataFrame(self.x_perturb)
        except AttributeError:
            _df1 = 0
        try:
            _df2 = pd.read_csv(
                os.path.join(
                    self.out_dir,
                    "x_perturb.tsv"),
                sep="\t", header=None)
        except FileNotFoundError:
            _df2 = 1
        if type(_df1) == int and type(_df2) == int:
            _ = "self.x_perturb and x_perturb.tsv are both not exist"
            logging.error(_)
            raise AttributeError(_)
        elif type(_df1) == pd.DataFrame and type(_df2) == pd.DataFrame:
            if not (abs(_df1.values - _df2.values) < 1e-6).all():
                logging.warning(f"self.x_perturb is not equal to x_perturb.tsv in {self.out_dir}")
                logging.warning(f"Maybe forget to use override=True in perturb_all()?")

        df2 = _df1 if isinstance(_df1, pd.DataFrame) else _df2


        # Extract the first column of 1.tsv and merge it with all columns of 2.tsv
        df2 = pd.concat([df1.iloc[:, 0], df2], axis=1)
        df3 = pd.DataFrame(np.vstack([df1.values, df2.values]))
        df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to 3.tsv
        df3.to_csv(
            os.path.join(_,f"{self.dataset}_TRAIN.tsv")
            , sep='\t', index=False, header=False)
