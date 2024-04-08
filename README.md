# Breast Cancer Risk Prediction

## Overview
This project aims to predict breast cancer risk using various machine learning models. The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The values represent characteristics of the cell nuclei present in the image. Through exploratory data analysis (EDA), data cleaning, and feature engineering, we train several models to classify and predict the diagnosis (malignant or benign).

## Installation

The code is written in Python and designed to be run in Google Colab. To run it locally, ensure you have the following prerequisites installed:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- XGBoost

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost
```

## Data Preparation

Before running the analysis, you need to download the dataset from the UCI Machine Learning Repository:

- Dataset Link: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))


## Usage

The notebook is divided into several sections:

1. **EDA and Data Cleaning**: Inspect the data and prepare it for modeling by handling missing values and unnecessary columns.
2. **Data Visualization**: Visualize the distribution and correlation of features.
3. **Data Preprocessing**: Scale features and split the dataset into training and test sets.
4. **Modeling**: Train and evaluate different models, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree and XGBoost. Functions are provided to report model performance and plot AUC curves.
5. **Feature Importance Analysis**: (Only for some models) Evaluate the importance of different features in the prediction.

To run the analysis, execute the notebook cells sequentially. You can tweak model parameters or try different models by modifying the relevant sections of the code.

## Contributing

Contributions to this project are welcome. You can contribute in several ways:

- Improving the EDA and feature engineering steps.
- Experimenting with different models or tuning the existing ones.
- Enhancing data visualization.
- Adding more detailed explanations of the data analysis process.

Please create a pull request or open an issue to discuss your proposed changes.
