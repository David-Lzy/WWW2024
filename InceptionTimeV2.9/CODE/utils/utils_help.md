# Documentation for `utils.py` in the Time Series Classification Project

## Overview
The `utils.py` script contains utility functions and classes that support various operations in the Time Series Classification project. These include data loading, preprocessing, metrics calculation, and other helper functions.

## Key Functions

### `readucr(filename, delimiter="\t")`
- **Description**: Reads a UCR dataset file.
- **Parameters**:
  - `filename`: Path to the UCR dataset file.
  - `delimiter`: Delimiter used in the dataset file, default is tab (`"\t"`).
- **Returns**: A tuple of `(X, Y)` where `X` is the time series data and `Y` is the labels.

### `map_label(y_data)`
- **Description**: Maps labels in `y_data` to a range of integer values.
- **Parameters**:
  - `y_data`: Array of labels.
- **Returns**: Mapped labels as a numpy array.

### `load_data(dataset, phase, batch_size, data_path)`
- **Description**: Loads a dataset and returns it as a PyTorch DataLoader.
- **Parameters**:
  - `dataset`: Name of the dataset.
  - `phase`: Dataset phase, either "TRAIN" or "TEST".
  - `batch_size`: Batch size for the DataLoader.
  - `data_path`: Path to the dataset.
- **Returns**: DataLoader, shape of the data, and number of classes.

### `data_loader(dataset, batch_size, data_path)`
- **Description**: Helper function to load both training and testing data.
- **Parameters**:
  - `dataset`: Name of the dataset.
  - `batch_size`: Batch size for the DataLoader.
  - `data_path`: Path to the dataset.
- **Returns**: Train and test DataLoaders, shapes, and number of classes.

### `metrics(targets, preds)`
- **Description**: Calculates precision, recall, and F1 score.
- **Parameters**:
  - `targets`: True labels.
  - `preds`: Predicted labels.
- **Returns**: Precision, recall, and F1 score.

### `create_directory(directory_name)`
- **Description**: Creates a directory if it does not exist.
- **Parameters**:
  - `directory_name`: Path of the directory to create.

### `save_metrics(directory_name, phase, metrics)`
- **Description**: Saves metrics to a CSV file.
- **Parameters**:
  - `directory_name`: Directory where to save the metrics.
  - `phase`: Phase of the training/testing ("train" or "test").
  - `metrics`: Dictionary of metrics to save.

### `concat_metrics_train(mode, method, datasets)`
- **Description**: Concatenates metrics for training from multiple datasets.
- **Parameters**:
  - `mode`: Mode of the operation, e.g., "train".
  - `method`: Method or approach used.
  - `datasets`: List of datasets.

### `concat_metrics_attack(method, datasets)`
- **Description**: Concatenates attack metrics from multiple datasets.
- **Parameters**:
  - `method`: Attack method used.
  - `datasets`: List of datasets.

### `write_attack_metrics_to_csv(csv_path, *args)`
- **Description**: Writes attack metrics to a CSV file.
- **Parameters**:
  - `csv_path`: Path to save the CSV file.
  - `*args`: Metrics to be written.

### `get_method_loc(methods)`
- **Description**: Generates a string representation of a method's configuration.
- **Parameters**:
  - `methods`: Dictionary or list of methods and their parameters.
- **Returns**: String representation of the method configuration.

### `build_defence_dict(defence, angle, Augment, adeversarial_training)`
- **Description**: Builds a dictionary for defense configurations.
- **Parameters**:
  - `defence`: Base defense configuration.
  - `angle`: Angle parameter.
  - `Augment`: Augmentation configurations.
  - `adeversarial_training`: Indicates if adversarial training is enabled.
- **Returns**: A dictionary with defense configurations.

### `determine_epochs(wanted_e, real_end_e, this_time_e, continue_train)`
- **Description**: Determines the range of epochs for training.
- **Parameters**:
  - `wanted_e`: Desired number of epochs.
  - `real_end_e`: Previously completed epochs.
  - `this_time_e`: Current epoch count.
  - `continue_train`: Flag to continue training.
- **Returns**: Tuple indicating the start and end epochs for training.

## Usage
These utility functions are integral to data handling, model training, and evaluation within the project. They should be imported and used as needed in various scripts of the project.

## Note
Ensure the correct file paths and dataset names are used when calling these functions. Also, modify and adapt these utilities as per the specific requirements of your project.
