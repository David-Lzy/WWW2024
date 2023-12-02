# Using the UCR Time Series Classification Archive

## Overview
The UCR Time Series Classification Archive is a comprehensive collection of time series datasets used widely for benchmarking time series classification algorithms. The archive was last significantly updated in Fall 2018 and contains datasets that span various domains such as ECG, sensor readings, images, motion, and more.

## Accessing the Archive
To access the datasets in the UCR Time Series Archive:

1. **Visit the Archive**: Go to [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

2. **Read the Briefing Document**: Before downloading the datasets, it's recommended to read the briefing document available in PDF or PowerPoint format on the archive page. This document also contains the password required for downloading the datasets.

3. **Download the Datasets**: The entire archive can be downloaded as a zipped file (approximately 260 MB). Ensure you have enough storage and a stable internet connection.

## Dataset Information
The archive comprises a wide range of datasets, each varying in the number of classes, the length of the time series, and the domain of application. Key information about each dataset includes:

- **Domain**: Classifies the dataset by its application domain (e.g., Image, Sensor, ECG).
- **Dataset Name**: The name of the dataset (e.g., Adiac, ArrowHead).
- **Number of Instances**: Number of time series samples in the dataset.
- **Number of Classes**: The number of distinct classes or categories in the dataset.
- **Length of Time Series**: The length of each time series in the dataset.

## Usage
To use these datasets:

1. **Download and Extract**: After downloading, extract the zipped file to a preferred directory.

2. **Read Dataset-Specific Information**: Each dataset may come with its own README or documentation. It's crucial to read these files to understand the dataset's specifics, like preprocessing steps or the significance of each class.

3. **Import into Your Project**: Depending on the programming language or tool you are using, import the datasets. For example, in Python, you can use libraries like `pandas` to read the datasets.

4. **Data Preprocessing**: Some datasets might require preprocessing, such as normalization or reshaping, before they can be used for training models.

5. **Model Training and Evaluation**: Use the datasets to train and evaluate your time series classification models.

## Citation
If you use data from the UCR Time Series Classification Archive in your research, please cite it as follows:

```
@misc{UCRArchive2018,
    title = {The UCR Time Series Classification Archive},
    author = {Dau, Hoang Anh and Keogh, Eamonn and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan 
              and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Yanping and Hu, Bing 
              and Begum, Nurjahan and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo, and Hexagon-ML},
    year = {2018},
    month = {October},
    note = {\url{https://www.cs.ucr.edu/~eamonn/time_series_data_2018/}}
}
```

For further details and specific questions about the datasets, it's advisable to refer to the original research papers or contact the dataset contributors.
