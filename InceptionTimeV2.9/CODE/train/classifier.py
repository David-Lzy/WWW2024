import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *
from CODE.train.Augmentation import Augmentation


class InceptionModule(nn.Module):
    """
    Represents an Inception block, a fundamental component of the Inception architecture for time series data.

    Parameters:
    - in_channels (int): Number of input channels to the module.
    - nb_filters (int): Number of filters in convolutional layers.
    - kernel_size (int): Kernel size for convolutional layers.
    - bottleneck_size (int, optional): Size of the bottleneck layer. Defaults to 32.
    - use_bottleneck (bool, optional): Whether to use a bottleneck layer. Defaults to True.

    The module consists of a set of convolutional layers with varying kernel sizes and a bottleneck layer
    (optional) to reduce dimensionality.
    """

    def __init__(
        self,
        in_channels,
        nb_filters,
        kernel_size,
        bottleneck_size=32,
        use_bottleneck=True,
    ):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_size, kernel_size=1, bias=False
            )
            conv_input_channels = bottleneck_size
            conv_mp_size = bottleneck_size * 4
        else:
            conv_input_channels = in_channels
            conv_mp_size = in_channels

        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad1d(padding=(k // 2 - (k + 1) % 2, k // 2), value=0),
                    nn.Conv1d(
                        conv_input_channels, nb_filters, kernel_size=k, bias=False
                    ),
                )
                for k in kernel_size_s
            ]
        )

        self.conv_mp = nn.Conv1d(conv_mp_size, nb_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        max_pool_1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=(3 - 1) // 2)

        if self.use_bottleneck and x.size(1) > 1:
            x = self.bottleneck(x) if hasattr(self, "bottleneck") else x

        conv_list = [conv(x) for conv in self.convs]

        conv_mp_out = self.conv_mp(max_pool_1)
        conv_list.append(conv_mp_out)

        x = torch.cat(conv_list, dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Classifier_INCEPTION(nn.Module):
    """
    Main classifier using multiple InceptionModule blocks. Designed for time series data classification.

    Parameters:
    - input_shape (tuple): Shape of the input data.
    - nb_classes (int, optional): Number of output classes. Defaults to 2.
    - nb_filters (int, optional): Number of filters for convolutional layers. Defaults to 32.
    - use_residual (bool, optional): Whether to use residual connections. Defaults to True.
    - use_bottleneck (bool, optional): Whether to use bottleneck layers in Inception modules. Defaults to True.
    - depth (int, optional): Number of InceptionModule blocks. Defaults to 6.
    - kernel_size (int, optional): Kernel size for convolutional layers in InceptionModule blocks. Defaults to 41.
    - defence (dict, optional): Dictionary specifying defense mechanisms. Defaults to an empty dictionary.

    The classifier aggregates features learned by the Inception modules and passes them through
    fully connected layers for classification.
        Usage:

        Importing the necessary components:
            from classifier import InceptionModule, InceptionClassifier

        Creating an Inception block:
            `inception_block = InceptionModule(in_channels=1, nb_filters=32, kernel_size=41)`

        Creating the Inception classifier:
            `classifier = InceptionClassifier(input_shape=(None, 1, 128), nb_classes=10)`

        Forward pass:
            output = classifier(input_tensor)

        Notes:
            Ensure that the Package module is accessible or available in the directory structure, as it's imported at the beginning of classifier.py.
            The softmax activation function is applied at the end of the InceptionClassifier forward pass, so the output will be in the range [0, 1] and sum up to 1 across classes.

    """

    def __init__(
        self,
        input_shape,
        nb_classes=2,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        defence={
            "angle": False,
            "get_w_mtx": False,
            "Augmentation": {
                "ahead_model": {},
                "in_model": {},
            },
        },
    ):

        self.forward = self.__forward_softmax__
        self.__f__ = self.__f_no_w__

        self.defence = defence
        if type(self.defence) == dict:
            if defence.get("angle", False):
                self.forward = self.__forward_angel___

            if defence.get("get_w_mtx", False):
                self.__f__ = self.__f_w_mtx__

            if defence.get("Augmentation", False):
                if defence["Augmentation"].get("ahead_model", False):
                    if len(defence["Augmentation"]["ahead_model"]) > 0:
                        self.__p_1__ = self.__AUG_1__
                if defence["Augmentation"].get("in_model", False):
                    if len(defence["Augmentation"]["in_model"]) > 0:
                        self.__p_2__ = self.__AUG_2__


        super(Classifier_INCEPTION, self).__init__()
        self.use_residual = use_residual
        self.depth = depth
        self.nb_filters = nb_filters

        # Define shortcut layers
        self.shortcut_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.nb_filters * 4 if i else 1,
                    self.nb_filters * 4,
                    kernel_size=1,
                    bias=False,
                )
                for i in range(depth // 3)
            ]
        )
        self.shortcut_bns = nn.ModuleList(
            [nn.BatchNorm1d(self.nb_filters * 4) for _ in range(depth // 3)]
        )
        self.shortcut_relus = nn.ModuleList([nn.ReLU() for _ in range(depth // 3)])

        # Adjust the in_channels according to the input shape
        #self.in_channels = input_shape[1]
        self.in_channels = 1
        self.layers = nn.ModuleList()
        # self.shortcut_layers = nn.ModuleList()

        for d in range(self.depth):
            self.layers.append(
                InceptionModule(
                    self.in_channels,
                    nb_filters,
                    kernel_size - 1,
                    use_bottleneck=True if d else False,
                )
            )
            self.in_channels = nb_filters * 4

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.in_channels, nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def short_cut_layer(self, x, input_res, shortcut_idx):
        x_ = self.shortcut_convs[shortcut_idx](input_res)
        x_ = self.shortcut_bns[shortcut_idx](x_)
        x_new = x + x_
        x_new = self.shortcut_relus[shortcut_idx](x_new)
        return x_new

    def __forward_softmax__(self, x):
        x = self.__p_1__(x)
        input_res = x
        shortcut_idx = 0

        for d, layer in enumerate(self.layers):
            x = layer(x)

            if self.use_residual and d % 3 == 2:
                x = self.short_cut_layer(x, input_res, shortcut_idx)
                shortcut_idx += 1
                input_res = x

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.__p_2__(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def __f_w_mtx__(self, a, b):
        return a, b

    def __f_no_w__(self, a, _):
        return a

    def __forward_angel___(self, x):
        x = self.__p_2__(x)
        input_res = x
        shortcut_idx = 0

        for d, layer in enumerate(self.layers):
            x = layer(x)

            if self.use_residual and d % 3 == 2:
                x = self.short_cut_layer(x, input_res, shortcut_idx)
                shortcut_idx += 1
                input_res = x

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.__AUG_2__(x)
        logits = self.fc(x)

        # path1
        y_pred = self.softmax(logits)

        # path2
        w = self.fc.weight
        z_norm = z.norm(dim=1, keepdim=True)
        w_norm = w.norm(dim=1, keepdim=True)

        norm_matrix = z_norm @ w_norm.T

        cos = logits / norm_matrix
        cos = ((cos + 1) / 2) ** 2

        # Prototype_trained
        w_mtx = (w @ w.T) / w_norm**2

        # return self.__f__(cos * y_pred, w_mtx)
        return self.__f__(cos * 1, w_mtx)
        # TODO:
        # There is no Augmentation for the moment,
        # So when angle is True, Augmentation in "in_model" will be ignored.

    def __AUG_1__(self, x):
        AUG = self.defence['Augmentation']
        for name, paramater in AUG['ahead_model'].items():
            _ = name.split('.')[1]
            if type(paramater) == dict:
                x = Augmentation.get_method()[_](x, **paramater)
            else:
                x = Augmentation.get_method()[_](x)
        return x

    def __AUG_2__(self, x):
        AUG = self.defence['Augmentation']
        for name, paramater in AUG['in_model'].items():
            _ = name.split('.')[1]
            if type(paramater) == dict:
                x = Augmentation.get_method()[_](x, **paramater)
            else:
                x = Augmentation.get_method()[_](x)
        return x

    def __p_1__(self, x):
        return x

    def __p_2__(self, x):
        return x