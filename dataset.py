import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import main


def get_row_dataset():
    return pd.read_csv(main.read_args().dataset_path_train, index_col=False)


def get_processed_dataset():
    df = get_row_dataset()

    handle_nan(df)  # Replace nan with better suited datas
    correct_data_types(df)  # Data types (object instead of int\float)
    convert_string2categorical(
        df)  # convert string categorical to int categorical -> now we can use the correlation matrix

    return df


def handle_nan(df: pd.DataFrame):
    #   replace '?' to 'Nan' so that 'pd.dropna' can remove the missing values
    df.replace("?", np.NaN, inplace=True)

    #   Data cleaning
    nan_ratios = df.isna().sum() / len(df)

    for index, value in nan_ratios.items():
        if value > 0 and index != "symboling":
            #   columns has too many nan(s) --> delete it
            if value > main.read_args().nan_tolerance:
                print(f"Column {index} has too much null values ---> deleted")
                df.drop(index, axis=1)

            #   columns of strings: can't use arithmetic average --> use most common value
            elif df[index].dtype == 'object':
                common_value = df[index].value_counts().idxmax()
                df[index].fillna(common_value, inplace=True)

            #   other columns --> replace nan with arithmetic average
            else:
                column_avg = df[index].astype("float").mean()
                df[index].fillna(column_avg, inplace=True)


def correct_data_types(df: pd.DataFrame):
    df[["horsepower"]] = df[["horsepower"]].astype("int")
    df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
    df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
    df[["price"]] = df[["price"]].astype("float")
    df[["peak-rpm"]] = df[["peak-rpm"]].astype("int")


def convert_string2categorical(df: pd.DataFrame):
    label_encoder = LabelEncoder()

    # Seleziona le colonne di tipo oggetto (stringhe) nel tuo DataFrame
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Applica la codifica alle colonne categoriche
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])


def class_instances(df: pd.DataFrame):
    print("label\tinstances")
    for i in range(-3, +4):
        cont = df['symboling'].value_counts().get(i)
        print(i, "\t", cont)


def enlarge(df: pd.DataFrame):
    selected_rows = df.loc[df['symboling'] == -2]  # select multiple rows
    duplicated_rows = pd.concat([selected_rows] * 3, ignore_index=True)  # duplica
    df = pd.concat([df, duplicated_rows], ignore_index=True)  # concatena

    return df


if __name__ == "__main__":
    df = get_processed_dataset()
    #df = enlarge(df)
    print(df)
    print(df.dtypes)
    class_instances(df)
