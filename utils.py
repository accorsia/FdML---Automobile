import pandas as pd
from sklearn import metrics


def dataset_info(df: pd.DataFrame):
    title("Dataset info after processing")

    print(">>> df.info() <<<")
    print(df.info())

    print("\n>>> df.describe() <<<")
    print(df.describe())

    print("\n>>> df.head() <<<")
    print(df.head())

    print("\n>>> df.columns() <<<")
    print(df.columns)

    print("\n>>> df.isnull().any()<<<")
    print(df.isnull().any())


def title(title):
    print('\n############ ', title, ' ############')


def print_ensemble_metrics(mse, mae, r2):
    title("Ensemble: training (mean)")

    print("Stacking ensemble - MSE: ", mse)
    print("Stacking ensemble - MAE: ", mae)
    print("Stacking ensemble - R2: ", r2)


def print_final_metrics(y_test, y_pred):
    title("Ensemble: validation")

    print('Final Regressor - MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('Final Regressor - MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('Final Regressor - R2:', metrics.r2_score(y_test, y_pred))
