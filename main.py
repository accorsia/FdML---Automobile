import argparse

import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dataset

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
                        default=5)
    #   ratio of nan values after which the relative column will be dropped
    parser.add_argument("--nan_tolerance",
                        type=float,
                        default=0.5)
    parser.add_argument("--train_test_ratio",
                        type=float,
                        default=0.2)

    return parser.parse_args()


def feature_selection(df: pandas.DataFrame, target_col_name):
    corr_matrix = df.corr(numeric_only=True).abs()  # 'abs()': needed to select the best features

    #   Plot correlation matrix
    """corr_matrix = df.corr()
    sea.heatmap(corr_matrix, annot=True)
    plt.show()"""

    nf = 20  # nuber of best features to extract

    # sort 'symboling' column from correlation matrix - select the first 20 values (best features) and remove the
    # first one ('symboling' itself)
    best_features_columns = corr_matrix[target_col_name].sort_values(ascending=False).iloc[1:nf+1]
    best_features_indexes = best_features_columns.index.values

    return best_features_indexes


if __name__ == "__main__":
    args = read_args()
    df = dataset.get_processed_dataset()  # get cleaned dataset

    #   Split dataframe into: Features, Target
    best_features = feature_selection(df, target_name)
    X = df[best_features].values
    Y = df[target_name].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.train_test_ratio, random_state=0)

    #   Scale data (apply training set fit to test set)
    scaler = StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.trasnform(x_test)


    print("end of main")
