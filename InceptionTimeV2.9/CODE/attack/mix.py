import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *


class Mix(Attack):
    """
    The `Mix` class, extending the `Attack` class, is designed to perform adversarial attacks
    using a combination of different attack methods on an Inception-based classifier for time series data.

    Initialization:
    mix_instance = Mix(
        dataset,
        Model=Classifier_INCEPTION,
        batch_size=64,
        epoch=1000,
        eps_init=0.001,
        eps=0.1,
        gamma=0.01,
        device=None,
        c=0.0001,
        swap=False,
        CW=False,
        kl_loss=False,
        swap_index=1
    )

    Parameters:
    - dataset (str): Name of the dataset being used.
    - Model (torch.nn.Module, optional): The classifier model to be attacked. Defaults to Classifier_INCEPTION.
    - batch_size (int, optional): Batch size for processing. Defaults to 64.
    - epoch (int, optional): Total number of epochs for the attack. Defaults to 1000.
    - eps_init (float, optional): Initial epsilon perturbation magnitude. Defaults to 0.001.
    - eps (float, optional): Maximum allowed perturbation. Defaults to 0.1.
    - gamma (float, optional): Parameter for adjusting the target. Defaults to 0.01.
    - device (torch.device, optional): Device for the attack (e.g., 'cuda' or 'cpu'). Defaults to None.
    - c (float, optional): Regularization parameter. Defaults to 0.0001.
    - swap (bool, optional): Whether to use the SWAP method. Defaults to False.
    - CW (bool, optional): Whether to use the Carlini-Wagner loss. Defaults to False.
    - kl_loss (bool, optional): Whether to use the Kullback-Leibler loss. Defaults to False.
    - swap_index (int, optional): Index for the SWAP method. Defaults to 1.

    Attributes:
    - method (str): Describes the attack method being used.
    - out_dir (str): Directory to save the attack results.

    Methods:
    - __get_method__(...): Determines the attack method based on the provided parameters.
    - __get_y_target_RAND__(...): Generates target labels using random target generation.
    - __get_y_target_SWAP__(...): Generates target labels using the SWAP method.
    - __cross_entropy_LOSS__(...): Computes the cross-entropy loss.
    - __kullback_leibler_LOSS__(...): Computes the Kullback-Leibler divergence loss.
    - __CW_loss_fun__(...): Computes the Carlini-Wagner loss for the attack.
    - __NoCW_loss_fun__(...): Computes the loss for non-Carlini-Wagner attack.
    - __perturb__(...): Performs the adversarial attack on the input.
    - __perturb_g__(...): Method to perturb input for gradient descent-based attack.
    - __perturb_s__(...): Method to perturb input for sign-based attack.

    Notes:
    - Ensure that necessary dependencies (Package, Classifier_INCEPTION, etc.) are accessible.
    - The `Mix` class offers a flexible framework for performing adversarial attacks with various methods.
    """

    def __init__(
        self,
        # parameter for Attacker
        dataset=None,
        Model=None,
        batch_size=None,
        epoch=None,
        eps=None,
        device=None,
        train_method_path=None, # know train_method pth location
        path_parameter=None, # know attack output location
        # parameter for swap
        swap=None,
        swap_index=None,
        gamma=None,
        # parameter for Kullback Leibler
        kl_loss=None,
        # parameter for Attack Gradient Descent
        CW=None,
        c=None,
        # parameter for init r
        eps_init=None,
        # parameter for BIM
        sign_only=None,
        alpha=None,
        
        adeversarial_training=None,

    ):
        # Call the parent class's constructor

        super(Mix, self).__init__(
            dataset=dataset,
            Model=Model,
            batch_size=batch_size,
            epoch=epoch,
            eps_init=eps_init,
            eps=eps,
            device=device,
            path_parameter=path_parameter,
            train_method_path=train_method_path,
            adeversarial_training=adeversarial_training,
            )

        if swap is not None:
            self.config['swap'] = swap
        if swap_index is not None:
            self.config['swap_index'] = swap_index
        if gamma is not None:
            self.config['gamma'] = gamma
        if kl_loss is not None:
            self.config['kl_loss'] = kl_loss
        if CW is not None:
            self.config['CW'] = CW
        if c is not None:
            self.config['c'] = c
        if sign_only is not None:
            self.config['sign_only'] = sign_only
        if alpha is not None:
            self.config['alpha'] = alpha



        if swap:
            self.config['swap'] = True
            self.swap_index = self.config['swap_index']
            self.gamma = self.config['gamma']
            self._get_y_target = self.__get_y_target_SWAP__
        else:
            self.config['swap'] = False
            self._get_y_target = self.__get_y_target_RAND__

        if kl_loss:
            self.config['kl_loss'] = True
            self.__LOSS__ = self.__kullback_leibler_LOSS__
        else:
            self.config['kl_loss'] = False
            self.__LOSS__ = self.__cross_entropy_LOSS__

        if CW:
            self.config['CW'] = True
            self.c = self.config['c']
            self._loss_function = self.__CW_loss_fun__
        else:
            self.config['CW'] = False
            self._loss_function = self.__NoCW_loss_fun__

        if sign_only:
            self.config['sign_only'] = True
            self.alpha = self.config['alpha']
            self.__perturb__ = self.__perturb_s__
        else:
            self.config['sign_only'] = False
            self.__perturb__ = self.__perturb_g__



        if path_parameter is not None:
            self.config["path_parameter"] = {
                key: value for key, value in self.config.items() if key in path_parameter}
        else:
            self.config["path_parameter"] = {
                key: value for key, value in self.config.items() if key in self.config["path_parameter"]}
            
        self.path_parameter = self.config["path_parameter"]
        self.attack_method_path = get_method_loc(self.path_parameter)
        self.out_dir = os.path.join(
            ATTACK_OUTPUT_path,
            self.train_method_path,
            self.attack_method_path,
            dataset,)




    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        return r

    def __get_y_target_RAND__(self, x, *args):
        if len(args) > 0:
            logging.warning("At least more than one Parameter unused!")
        with torch.no_grad():
            y_pred = self.f(x)
            y_target = torch.zeros_like(y_pred)
            _, c = torch.max(y_pred, dim=1)
            for i in range(len(y_pred)):
                c_s = torch.arange(y_pred.shape[1], device=y_pred.device)
                c_s = c_s[c_s != c[i]]
                new_c = c_s[torch.randint(0, len(c_s), (1,))]
                y_target[i, new_c] = 1.0
            # y_target = torch.argmax(y_target, dim=1)
        return y_target, c

    def __get_y_target_SWAP__(self, x):
        # 这里index用于挑选那一个预测用于swap
        with torch.no_grad():
            y_pred = self.f(x)
            _, top2_indices = torch.topk(y_pred, self.swap_index + 1, dim=1)
            y_target = y_pred.clone()

            for i in range(len(y_pred)):
                c_top2 = top2_indices[i]
                mean_ = (
                    y_pred[i, c_top2[0]] + y_pred[i, c_top2[self.swap_index]]
                ) / 2  # 交换第一和第二项的值，保持原有分布
                y_target[i, c_top2[self.swap_index]] = mean_ + self.gamma
                y_target[i, c_top2[0]] = mean_ - self.gamma  # 让原始的第二项比第一项稍大一点
        return y_target, top2_indices[:, 0]

    def __cross_entropy_LOSS__(self, y_pred_adv, y_target):
        return nn.functional.cross_entropy(y_pred_adv, y_target, reduction="none")

    def __kullback_leibler_LOSS__(self, y_pred_adv, y_target):
        return nn.functional.kl_div(torch.log(y_pred_adv), y_target, reduction="none")

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i] == top1_index[i]:
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the L2 regularization
        l2_reg = torch.norm(r, p=2)

        return l2_reg * self.c + loss.mean()

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        return self.__LOSS__(y_pred_adv, y_target).mean()

    def __perturb_g__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        # 这里看起来不需要to_device
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            loss = self._loss_function(x, r, y_target, top1_index)
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

    def __perturb_s__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        sum_losses = np.zeros(self.epoch)

        for epoch in range(self.epoch):
            loss = self._loss_function(x, r, y_target, top1_index)
            optimizer.zero_grad()
            loss.backward()

            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.alpha * grad_sign
            # alpha is your step size for BIM
            r.data = torch.clamp(r.data, -self.eps, self.eps)

            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.info(f"Epoch: {epoch+1}/{self.epoch}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses
