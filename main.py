import argparse

import pandas
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
    best_features_columns = corr_matrix[target_col_name].sort_values(ascending=False).iloc[1:nf + 1]
    best_features_indexes = best_features_columns.index.values

    return best_features_indexes


def split_data(df: pandas.DataFrame):
    best_features = feature_selection(df, target_name)
    X = df[best_features].values
    Y = df[target_name].values

    return X, Y


def scale_data(x_train, x_test):
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
    models, models_names, models_hparametes = create_models_items()

    best_hparameters = []  # calculated best hparam value for each model
    estimators = []  # list of models with theirs metadata

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        reg = GridSearchCV(estimator=model, param_grid=hparameters, scoring='r2',
                           cv=read_args().cv)  # model data structure
        reg.fit(x_train, y_train)  # train model

        #   append model' s datas to data collector objects
        best_hparameters.append(reg.best_params_)
        estimators.append((model_name, reg))

        #   debug
        print('\n', model_name)
        print('R2 Score:', reg.best_score_)

    # print('############ Ensemble ############\n')

    """
    Solitamente si usa il modello più stabile com 'final_estimator', che non per forza deve essere il modello con R2 più alto
    In questo caso, 'RandomForestRegressor' è anche il modello con R2 più alto: ~0.75 vs ~0.55 degli altri modelli
    """
    ensemble_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
    return ensemble_model


if __name__ == "__main__":
    args = read_args()
    df = dataset.get_processed_dataset()  # get processed dataset: null values, data types, string categorical

    #   Split data housing
    X, Y = split_data(df)

    #   Split housings into dataframes
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.train_test_ratio, random_state=0)

    #   Scale dataframes (apply training set fit to test set)
    x_train, x_test = scale_data(x_train, x_test)

    #   Create object representing the ensemble of the models
    regressor_ensemble = create_ensemble(x_train, y_train)
    print(regressor_ensemble)





    print("end of main")
