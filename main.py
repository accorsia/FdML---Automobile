import argparse

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import dataset
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
                        default=13)

    return parser.parse_args()


def project_info():
    title("Project args")
    print("Cross validation foldings:\t", args.cv)
    print("Best features:\t", args.n_features)
    print("########################\n")


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


def split_data(df: pandas.DataFrame):
    best_features = feature_selection(df, target_name)

    X = df[best_features].values
    Y = df[target_name].values

    return X, Y


def scale_data(x_train, x_test):
    #   Apply 'train' scaler also to test set
    scaler = StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def create_models_items():
    models = [
        LinearRegression(),
        Ridge(),
        SVR(),
        RandomForestRegressor()
    ]

    models_names = ['Linear Regression', 'Ridge', 'SVR', 'Random Forest']

    models_hparametes = [
        {},  # Linear regression
        {'alpha': [0.1, 1.0, 10.0]},  # Ridge
        {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]},  # SVR
        {'n_estimators': [100, 200, 500]}  # Random Forest
    ]

    return models, models_names, models_hparametes


def create_ensemble(x_train, y_train):
    title("Regressors")

    models, models_names, models_hparametes = create_models_items()

    best_hparameters = []  # calculated best hparam value for each model
    estimators = []  # list of models with theirs metadata

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        #   'GridSearchCV': data structure, with models full info
        reg = GridSearchCV(estimator=model, param_grid=hparameters, scoring='r2', cv=read_args().cv)
        reg.fit(x_train, y_train)

        #   append created data structures to collector objects
        best_hparameters.append(reg.best_params_)
        estimators.append((model_name, reg))

        #   debug
        print(model_name)
        print('R2 Score:', reg.best_score_, "\n")

    """
    Solitamente si usa il modello più stabile com 'final_estimator', che non per forza deve essere il modello con R2 più alto
    In questo caso, 'RandomForestRegressor' è anche il modello con R2 più alto: ~0.75 vs ~0.55 degli altri modelli
    """
    ensemble_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
    return ensemble_model


def calculate_scores(ensemble, x_train, y_train):
    scores = cross_validate(ensemble, x_train, y_train, cv=args.cv,
                            scoring=('neg_mean_squared_error',
                                     'neg_mean_absolute_error',
                                     'r2'))

    mse_scores = -scores['test_neg_mean_squared_error']
    mae_scores = -scores['test_neg_mean_absolute_error']
    r2_scores = scores['test_r2']

    return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)


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

    #   Processed dataset: null values, data types, string categorical
    df = dataset.get_processed_dataset()

    #   Show dataset
    dataset_info(df)
    project_info()

    X, Y = split_data(df)  # split data housing
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.train_test_ratio, random_state=0)
    x_train, x_test = scale_data(x_train, x_test)  # scale dataframes (apply training set fit to test set also)

    #   Create ensemble
    reg_ensemble = create_ensemble(x_train, y_train)

    #   Ensemble scores
    mse, mae, r2 = calculate_scores(reg_ensemble, x_train, y_train)  # ensemble metrics

    print_ensemble_metrics(mse, mae, r2)  # print metrics

    #   Modello finale
    final_reg = reg_ensemble
    final_reg.fit(x_train, y_train)
    y_pred = final_reg.predict(x_test)
    categorize_prediction(y_pred)  # convert continuous values to categorical [-3,...,+3]

    print_final_metrics(y_test, y_pred)  # print metrics

    #   Plot
    plt.plot(np.linspace(0, 10, len(y_pred)), y_pred)
    plt.plot(np.linspace(0, 10, len(y_test)), y_test)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Comparison of True and Predicted Values')
    plt.legend()
    plt.show()
