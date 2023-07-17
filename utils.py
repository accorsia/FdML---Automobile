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


def project_info(args):
    title("Project args")
    print("Cross validation foldings:\t", args.cv)
    print("Best features:\t", args.n_features)
    print("########################################\n")
