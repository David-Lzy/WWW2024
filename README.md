# WWW2024
This is the repository about the robustness investigation in time series classification model by using grand based attack.


# InceptionTimeV2.9

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
   git clone [repository link]
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

## Customizing Training and Attacks

You can customize training parameters and attack strategies by modifying the JSON configuration files in the `CODE/config` directory.

## Contribution Guidelines

Contributions to the project are welcome. Please see `CONTRIBUTING.md` for how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please contact the [project maintainer](mailto:your-email@example.com).
