import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils
from arguments import read_args
from utils import title

args = read_args()


def my_create_classification_models_items():
    models = [
        SVC(),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    models_names = ['SVC', 'Random Forest Classifier', 'Gradient Boosting Classifier']

    models_hparametes = [
        {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]},  # SVC
        {'n_estimators': [100, 200, 500]},  # Random Forest Classifier
        {}  # GradientBoostingClassifier
    ]

    return models, models_names, models_hparametes


def final_create_classification_models_items():
    models = [
        KNeighborsClassifier(weights='distance'),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    models_names = ['KNN', 'Random Forest Classifier', 'Gradient Boosting Classifier']

    models_hparametes = [
        {'n_neighbors': list(range(1, 10, 2))},  # KNN
        {'n_estimators': [100, 200, 500]},  # Random Forest Classifier
        {}  # GradientBoostingClassifier
    ]

    return models, models_names, models_hparametes


def prj_create_classification_models_items():
    models = [
        KNeighborsClassifier(weights='distance'),
        SVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced')
    ]

    models_names = ['K-NN', 'SVC', 'DT']

    models_hparametes = [
        {'n_neighbors': list(range(1, 10, 2))},  # KNN
        {'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']},  # SVC
        {'criterion': ['gini', 'entropy']},  # DT
    ]

    return models, models_names, models_hparametes


def create_classification_ensemble(x_train, y_train):
    title("Classifiers")

    # models, models_names, models_hparametes = my_create_classification_models_items()
    # models, models_names, models_hparametes = prj_create_classification_models_items()
    models, models_names, models_hparametes = final_create_classification_models_items()

    best_hparameters = []
    estimators = []

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=args.cv)
        clf.fit(x_train, y_train)

        best_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf))

        print(model_name)
        print('Accuracy:', clf.best_score_, "\n")

    ensemble_classifier = StackingClassifier(estimators=estimators, final_estimator=KNeighborsClassifier())
    return ensemble_classifier


"""def calculate_classification_scores_weight(ensemble, x_train, y_train):
    title("Calculating classification scores: Accuracy, F1 weighted")
    scores = cross_validate(ensemble, x_train, y_train, cv=args.cv,
                            scoring=('accuracy', 'f1_weighted', 'f1_weighted', 'precision_weighted'))

    accuracy_scores = scores['test_accuracy']
    precision_scores = scores['test_precision_weighted']
    recall_scores = scores['test_recall_weighted']
    f1_scores = scores['test_f1_weighted']

    return np.mean(accuracy_scores), np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)"""


def calculate_classification_scores(ensemble, x_train, y_train, metrics):
    # title("Calculating classification scores: " + str(metrics))
    skf = StratifiedKFold(n_splits=args.cv)
    scores = cross_validate(ensemble, x_train, y_train, cv=skf, scoring=metrics)

    mean_scores = {}
    for metric in metrics:
        metric_scores = scores['test_' + metric]
        mean_scores[metric] = np.mean(metric_scores)

    return mean_scores


def print_metrics(metrics):
    utils.title("[Classification] training")

    for metric, value in metrics.items():
        print(f'- {metric.capitalize()}:\t{value}')


def print_final_metrics(y_test, y_pred):
    utils.title("[Classification] validation")

    print('Accuracy is ', accuracy_score(y_test, y_pred))
    print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))
