import numpy as np
from sklearn import metrics
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.svm import SVR

import utils
from arguments import read_args

args = read_args()


def create_regression_models_items():
    models = [
        # LinearRegression(),
        # Ridge(),
        SVR(),
        RandomForestRegressor(),
        GradientBoostingRegressor()
    ]

    models_names = ['SVR', 'Random Forest', 'Gradient Boosting Regressor']

    models_hparametes = [
        # {},  # Linear regression
        # {'alpha': [0.1, 1.0, 10.0]},  # Ridge
        {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]},  # SVR
        {'n_estimators': [100, 200, 500]},  # Random Forest
        {}  # GradientBoostingRegressor
    ]

    return models, models_names, models_hparametes


def create_regression_ensemble(x_train, y_train):
    utils.title("Regressors")

    models, models_names, models_hparametes = create_regression_models_items()

    best_hparameters = []  # calculated best hparam value for each model
    estimators = []  # list of models with theirs metadata

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        #   'GridSearchCV': data structure, with models full info
        reg = GridSearchCV(estimator=model, param_grid=hparameters, scoring='r2', cv=args.cv)
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


def calculate_regression_scores(ensemble, x_train, y_train):
    skf = StratifiedKFold(n_splits=args.cv)
    scores = cross_validate(ensemble, x_train, y_train, cv=skf,
                            scoring=('neg_mean_squared_error',
                                     'neg_mean_absolute_error',
                                     'r2'))

    mse_scores = -scores['test_neg_mean_squared_error']
    mae_scores = -scores['test_neg_mean_absolute_error']
    r2_scores = scores['test_r2']

    return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)


def print_metrics(mse, mae, r2):
    utils.title("[Regressor] Training")

    print("- MSE: ", mse)
    print("- MAE: ", mae)
    print("- R2: ", r2)

def print_final_metrics(y_test, y_pred):
    utils.title("[Regressor] validation")

    print('- MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('- MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('- R2:', metrics.r2_score(y_test, y_pred))
