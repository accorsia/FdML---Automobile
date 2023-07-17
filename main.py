import argparse

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import clf
import dataset
import reg
from utils import *

target_name = "symboling"


def read_args():
    parser = argparse.ArgumentParser(description='Processing input dataset')
    #   dataset file location path
    parser.add_argument("--dataset_path_train",
                        type=str,
                        default="dataset/imports-85.data",
                        help="path to the file containing the dataset")
    #   cross validation foldings
    parser.add_argument("--cv",
                        type=int,
                        default=2)
    #   ratio of nan values after which the relative column will be dropped
    parser.add_argument("--nan_tolerance",
                        type=float,
                        default=0.5)
    #   ratio of train \ test over the whole data-housing
    parser.add_argument("--train_test_ratio",
                        type=float,
                        default=0.2)
    #   how many best features (extracted from the correlation matrix)
    parser.add_argument("--n_features",
                        type=int,
                        default=10)

    return parser.parse_args()


def feature_selection(df: pandas.DataFrame, target_col_name):
    corr_matrix = df.corr(numeric_only=True).abs()  # 'abs()': needed to select the best features

    #   Plot correlation matrix
    """corr_matrix = df.corr()
    sea.heatmap(corr_matrix, annot=True)
    plt.show()"""

    nf = args.n_features  # nuber of best features to extract

    # sort 'symboling' column from correlation matrix - select the first 20 values (best features) and remove the
    # first one ('symboling' itself)
    best_features_columns = corr_matrix[target_col_name].sort_values(ascending=False).iloc[1:nf + 1]
    best_features_indexes = best_features_columns.index.values

    return best_features_indexes


def split_housing(df: pandas.DataFrame):
    best_features = feature_selection(df, target_name)

    X = df[best_features].values
    Y = df[target_name].values

    return X, Y


def scale_data(x_train, x_test):
    #  'Standard': media=0, deviazione=1
    scaler = StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)  # apply 'train' scaler also to test set

    return x_train, x_test


def categorize_prediction(y):
    for i in range(len(y)):
        if y[i] < -2.5:
            y[i] = -3
        elif y[i] > 2.5:
            y[i] = 3
        else:
            y[i] = np.rint(y[i])


if __name__ == "__main__":
    args = read_args()

    ###     Processed dataset: null values, data types, string categorical
    df = dataset.get_processed_dataset()

    ###     Info (debug)
    dataset_info(df)
    project_info(args)

    ###     Data split + scaling
    X, Y = split_housing(df)  # split data housing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.train_test_ratio, random_state=0)
    x_train, x_test = scale_data(x_train, x_test)

    ###     Create ensembles
    reg_ensemble = reg.create_regression_ensemble(x_train, y_train)
    clf_ensemble = clf.create_classification_ensemble(x_train, y_train)

    ###     Ensembles scores - calculate
    mse, mae, r2 = reg.calculate_regression_scores(reg_ensemble, x_train, y_train)
    accuracy, precision, recall, f1 = clf.calculate_classification_scores(clf_ensemble, x_train, y_train)

    ###     Ensembles scores - print
    reg.print_metrics(mse, mae, r2)
    clf.print_metrics(accuracy, precision, recall, f1)

    ###     Final regressor
    final_reg = reg_ensemble
    final_reg.fit(x_train, y_train)
    y_reg_pred = final_reg.predict(x_test)
    categorize_prediction(y_reg_pred)  # convert continuous values to categorical [-3,...,+3]

    reg.print_final_metrics(y_test, y_reg_pred)  # print final metrics

    ###     Final classifier
    final_clf = clf_ensemble
    final_clf.fit(x_train, y_train)
    y_clf_pred = final_clf.predict(x_test)

    clf.print_final_metrics(y_test, y_clf_pred)



    #   Plot
    """plt.plot(np.linspace(0, 10, len(y_pred)), y_pred)
    plt.plot(np.linspace(0, 10, len(y_test)), y_test)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Comparison of True and Predicted Values')
    plt.legend()
    plt.show()"""
