import argparse


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
