import numpy as np
import pandas
import seaborn as sea
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import clf
import dataset
import graph
import reg
from arguments import read_args
from utils import *

target_name = "symboling"


def feature_selection(df: pandas.DataFrame, target_col_name):
    corr_matrix = df.corr(numeric_only=True).abs()  # 'abs()': needed to select the best features

    #   Plot correlation matrix
    sea.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".1f")
    plt.show()

    nf = args.n_features  # nuber of best features to extract

    # sort 'symboling' column from correlation matrix - select the first 20 values (best features) and remove the
    # first one ('symboling' itself)
    best_features_columns = corr_matrix[target_col_name].sort_values(ascending=False).iloc[1:nf + 1]
    best_features_indexes = best_features_columns.index.values

    graph.plot_correlation(corr_matrix[target_col_name].sort_values(ascending=False).iloc[1:])

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


def calculate_common_metric(y_reg_pred, y_clf_pred, y_test):
    title("Custom metric")

    if len(y_clf_pred) == len(y_reg_pred) and len(y_reg_pred) == len(y_test):
        print("Correct y(s) length\n")
        clf_right = 0
        reg_right = 0

        for i in range(len(y_test)):
            if y_clf_pred[i] == y_test[i]:
                clf_right += 1
            if y_reg_pred[i] == y_test[i]:
                reg_right += 1

        perc_reg = reg_right / len(y_test)
        perc_clf = clf_right / len(y_test)

        print("- [Regression] rights = ", reg_right, "/", len(y_test))
        print("-> [Regression] accuracy = ", perc_reg)
        print("-> [Regression] score (+1 right, -1 wrong) = ", reg_right - (len(y_test) - reg_right), "/", len(y_test))

        print("\n- [Classification] rights = ", clf_right, "/", len(y_test))
        print("-> [Classification] accuracy = ", perc_clf)
        print("-> [Classification] score (+1 right, -1 wrong) = ", clf_right - (len(y_test) - clf_right), "/", len(y_test))
    else:
        print("--- Wrong y(s) length! ---")

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
    df = dataset.enlarge(df)  # duplicate records of class which are poorly represented -> avoid cv problems

    ###     Info (debug)
    # dataset_info(df)
    project_info(args)

    ###     Data split + scaling
    X, Y = split_housing(df)  # split data housing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.train_test_ratio, random_state=42)
    x_train, x_test = scale_data(x_train, x_test)


    ###     Create ensembles
    reg_ensemble = reg.create_regression_ensemble(x_train, y_train)
    clf_ensemble = clf.create_classification_ensemble(x_train, y_train)

    ###     Ensembles training scores - calculate
    mse, mae, r2 = reg.calculate_regression_scores(reg_ensemble, x_train, y_train)
    clf_scores = clf.calculate_classification_scores(clf_ensemble, x_train, y_train,
                                                     ['accuracy',
                                                      'f1_weighted',
                                                      'precision_weighted',
                                                      'recall_weighted'])

    ###     Ensembles training scores - print
    reg.print_metrics(mse, mae, r2)
    clf.print_metrics(clf_scores)

    ###     Final regressor
    final_reg = reg_ensemble
    final_reg.fit(x_train, y_train)
    y_reg_pred = final_reg.predict(x_test)
    categorize_prediction(y_reg_pred)  # convert continuous values to categorical [-3,...,+3]

    ###     Final classifier
    final_clf = clf_ensemble
    final_clf.fit(x_train, y_train)
    y_clf_pred = final_clf.predict(x_test)

    ###     Final metrics
    reg.print_final_metrics(y_test, y_reg_pred)
    clf.print_final_metrics(y_test, y_clf_pred)

    ###     Custom common metric
    calculate_common_metric(y_reg_pred, y_clf_pred, y_test)

    #   Serialize
    np.savez('y_values.npz', y_test=y_test, y_clf_pred=y_clf_pred, y_reg_pred=y_reg_pred)

    #   Plot
    graph.plot_prediction_errors(y_test, y_clf_pred, y_reg_pred)
    graph.plot_prediction_differences(y_clf_pred, y_reg_pred)
    graph.plot_error_by_class(y_test, y_clf_pred, y_reg_pred)

    space = np.linspace(0, len(y_test), len(y_test))
    plt.plot(space, y_test)
    plt.plot(space, y_clf_pred)
    plt.plot(space, y_reg_pred)
    plt.xlabel('Samples')
    plt.ylabel('Prediction')
    plt.title('Comparison of Regressions - Classification - Ground Truth')
    plt.legend()
    plt.show()
