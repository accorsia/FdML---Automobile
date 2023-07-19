import os

import numpy as np
import pandas as pd


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
    print("\n------------ Project args --------------")
    print("Cross validation foldings:\t", args.cv)
    print("Best features:\t", args.n_features)
    print("----------------------------------------\n")

def serialize(filepath, *args, **kwds):
    directory = os.path.dirname(filepath)

    # Crea le cartelle intermedie se non esistono
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Rimuove il file se esiste già
    if os.path.exists(filepath):
        os.remove(filepath)

    # Salva i dati nel file npz
    np.savez(filepath, *args, **kwds)
