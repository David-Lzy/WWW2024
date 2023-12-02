import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Package import *


class Augmentation():
    """
    The `Augmentation` class provides a collection of static methods for data augmentation,
    particularly useful for time series data processing.

    It includes methods such as jittering, masking, and adding noise, which can be used to
    artificially augment dataset size and variability.

    Methods:
    - Jitter: Adds random jitter to the data.
    - binomial_mask: Applies a binomial mask for random data deletion.
    - continuous_mask: Applies a continuous mask to a portion of the data.
    - gaussian_noise: Adds Gaussian noise to the data.
    - gaussian_smooth: Applies Gaussian smoothing to the data.
    - nothing: Returns the data without any modifications.
    - get_method: Returns a dictionary of all available augmentation methods.
    - get_index: Returns a list of the names of all available augmentation methods.

    Usage Example:
    ```
    augmented_data = Augmentation.Jitter(data, p=0.5, amplitude=0.2)
    ```
    """

    Avoid_name = [
        'get_method',
        'get_index'
    ]

    def __init__(self) -> None:
        self.methods = {
        name: method
        for name, method in inspect.getmembers(self, predicate=inspect.isfunction)
        # if isinstance(self.__dict__.get(name, None), staticmethod)
    }

    @staticmethod
    def Jitter(x, p=0.5, amplitude=0.2):

        sign_array = (torch.randint(0, 2, size=x.shape) * 2 - 1).to(x.device)
        binary_array = (torch.rand(x.shape) < p).float().to(x.device)
        spike_array = sign_array * binary_array * amplitude
        x_jitter = x + spike_array

        return x_jitter.to(x.device)

    @staticmethod
    def binomial_mask(x, keep_prob=0.75):
        mask = torch.from_numpy(np.random.binomial(1, keep_prob, size=x.shape)).to(torch.bool)
        masked_x = x * mask.float().to(x.device)

        return masked_x

    @staticmethod
    def continuous_mask(
        x,
        max_chunk_ratio = 0.05,
        overall_mask_ratio = 0.25,
        ):

        length = x.shape[-1]
        max_mask_length = int(max_chunk_ratio*length)
        total_mask_length = int(overall_mask_ratio*length)

        masked_arr = x.clone()
        current_mask_length = 0

        while current_mask_length < total_mask_length:
            start = torch.randint(0, length, (1,)).item()
            mask_length = torch.randint(1, max_mask_length + 1, (1,)).item()

            if start + mask_length > length:
                mask_length = length - start
            if current_mask_length + mask_length > total_mask_length:
                mask_length = total_mask_length - current_mask_length
            if len(x.shape) ==3:
                masked_arr[:, :, start:start + mask_length] = 0
            else:
                masked_arr[:, start:start + mask_length] = 0

            current_mask_length += mask_length


        return masked_arr

    @staticmethod
    def gaussian_noise(x, mean=0, std = 0.1):

        size = x.shape
        mean_tensor = torch.full(size, mean).float().to(x.device)
        noise_array = torch.normal(mean_tensor, std).to(x.device)
        noised_x = x + noise_array

        return noised_x

    @staticmethod
    def gaussian_smooth(x, kernel_size=10, sigma=5):
        # Make sure x has the shape [batch_size, 1, sample_length]
        assert len(x.shape) == 3 and x.shape[1] == 1, "Expected input shape: [batch_size, 1, sample_length]"

        # Create a Gaussian kernel
        gauss_kernel = torch.exp(-torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()  # Normalize

        # Adapt to the input dimension
        gauss_kernel = gauss_kernel.view(1, 1, -1).to(x.device)

        padding_size = kernel_size // 2
        # Calculate the mean of the first and last padding_size elements
        start_mean = x[:, :, :padding_size].mean(dim=-1, keepdim=True)
        end_mean = x[:, :, -padding_size:].mean(dim=-1, keepdim=True)

        # Create padding tensors
        start_padding = start_mean.expand(-1, 1, padding_size)
        end_padding = end_mean.expand(-1, 1, padding_size)

        # Concatenate padding tensors with the input tensor x
        padded_x = torch.cat([start_padding, x, end_padding], dim=-1)

        # Perform convolution with padding set to 0, as we have manually added padding
        smoothed_x = F.conv1d(padded_x, gauss_kernel, padding=0)

        # Adjust the length of smoothed_x
        smoothed_x = 0.5 * (smoothed_x[:, :, :-1] + smoothed_x[:, :, 1:])

        return smoothed_x

    @staticmethod
    def nothing(x):
        return x

    @staticmethod
    def get_method(model = None):
        if model == None:
            model = Augmentation
        return {
            name: method
            for name, method in inspect.getmembers(model, predicate=inspect.isfunction)
            if isinstance(model.__dict__.get(name, None), staticmethod) and (not name in Augmentation.Avoid_name)
    }

    @staticmethod
    def get_index():
        return list(Augmentation.get_method().keys())

if __name__ == "__main__":
    print(Augmentation.get_method())
