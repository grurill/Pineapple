# Feature Engineering Application

This is a Feature Engineering Application built using PyQt5 and various machine learning libraries. It allows users to load CSV files, apply different feature engineering techniques, and save the processed data. The application supports operations such as handling missing values, encoding categorical variables, scaling features, binning, transformations, and feature selection.

## Features

- Load CSV: Load a CSV file and display its columns.
- Handle Missing Values: Apply strategies like dropping missing values, filling with mean, median, mode, or a custom value.
- Encoding: Perform label encoding or one-hot encoding on categorical columns.
- Scaling: Scale numerical features using StandardScaler or MinMaxScaler.
- Binning: Bin numerical features using equal width or equal frequency binning.
- Transformation: Apply transformations like log, square root, or Box-Cox.
- Feature Selection: Select the best features using the SelectKBest method based on ANOVA F-value.
- Save CSV: Save the processed data to a new CSV file.

## Prerequisites

Make sure you have Python 3.x installed. You also need to install the required packages:

```bash
pip install pandas numpy PyQt5 scikit-learn scipy
