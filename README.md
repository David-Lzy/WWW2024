# WWW2024
This is the repository about the robustness investigation in time series classification model by using grand based attack.

InceptionTimeV2.9 is a deep learning-based time series classification and adversarial training framework, focusing on training and attacking Inception networks. It offers a suite of tools for building, training, and evaluating time series models, and performing adversarial attacks to test model robustness.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Customizing Training and Attacks](#customizing-training-and-attacks)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Contact](#contact)

## Installation

To install and run InceptionTimeV2.9, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/David-Lzy/WWW2024#installation
   cd InceptionTimeV2.9
   ```

2. Install dependencies:
   ```
   pip install -r ENV/requirements.txt
   ```

3. Initialize the environment (optional):
   ```
   sh ENV/init_env.sh
   ```

## Getting Started

To get started with InceptionTimeV2.9, you can run the demo Jupyter notebooks to understand how to train models and perform attacks:

```
jupyter notebook CODE/demo/train.ipynb
jupyter notebook CODE/demo/attack.ipynb
```

## Project Structure

The project directory structure is as follows:

```
InceptionTimeV2.9
├── CODE
│   ├── attack           # Scripts related to attacks
│   ├── config           # Configuration files
│   ├── demo             # Demonstration Jupyter notebooks
│   ├── train            # Scripts related to training
│   └── utils            # Utilities and helper functions
├── DATA                 # Dataset directory
├── ENV                  # Environment configuration and dependencies
└── LOG                  # Log files
```

## Usage

The project includes various scripts and Jupyter notebooks for demonstration and performing different tasks. You can find different demonstration notebooks in the `CODE/demo` directory.

## Customizing Training and Attack Strategies

InceptionTimeV2.9 offers flexible customization options for both training and attack strategies. Here's a guide on how to navigate and utilize these features effectively:

### Configuration Files
- **Training and Attack Parameters**: The project includes JSON configuration files for detailed setup of training and attack parameters. You can find these configurations in the `CODE/config` directory. Specifically, `full_pramater_attack.jsonc` and `full_pramater_train.jsonc` contain exhaustive lists of parameters you can adjust to fit your needs.

### Customizing Attack Methods
- **DIY Attack Strategies**: If you wish to craft your own attack methodologies, you can modify `mix.py` located in the `CODE/attack` directory. This script is pivotal for defining and implementing custom attack logic.

### Data Augmentation
- **Enhancing Data**: To introduce or alter data augmentation methods, edit `InceptionTimeV2.9/CODE/train/Augmentation.py`. This file is crucial for applying various transformations and augmentations to your dataset, enhancing the robustness and variability of your training process.

### Shell Scripts for Execution
- **Automated Script Execution**: For ease of use, the project provides shell scripts that can be run to initiate training and attack processes with predefined or customized settings.
  - **Training Script (`train.sh`)**: Located in `InceptionTimeV2.9/CODE/shells`, this script can be used to start the training process. An example command in the script like `python headless_train.py --batch_size 256 --epoch 1500 --override True` initiates training with specific batch size, epoch count, and an override option.
  - **Attack Script (`attack.sh`)**: This script allows you to execute various attack scenarios with different configurations. For instance, commands within the script define batch size, epoch, swapping strategy, whether to use the Carlini-Wagner loss (CW), Kullback-Leibler loss (kl_loss), and other parameters. Users can customize these commands or add new ones to experiment with different attack strategies.

### Tips for Customization
- Always back up the original configuration files and scripts before making changes.
- Test your custom configurations with a small dataset or for a few epochs initially to validate the settings.
- Document any changes made to configurations or scripts for future reference or for sharing with the community.

This flexible structure allows both novice and advanced users to tailor the InceptionTimeV2.9 project according to their specific research needs or experimentation curiosity. Feel free to dive into these files and scripts to explore the full potential of the project.

## Contribution Guidelines

<details>
  <summary>
    Contributions to the project are welcome. Please see “CONTRIBUTING.md” for how to contribute 
  </summary>
   (It is not open yet, you need to wait until the article is received).
</details>

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contact

For any questions or suggestions, please contact the [project maintainer](mailto:cdong1997@gmail.com).
