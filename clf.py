import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import reg
import utils
from utils import title

cv = reg.ml_cv


def create_classification_models_items():
    models = [
        LogisticRegression(penalty='l2'),
        SVC(),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    models_names = ['Logistic Regression', 'SVC', 'Random Forest Classifier', 'Gradient Boosting Classifier']

    models_hparametes = [
        {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]},  # Logistic Regression
        {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0]},  # SVC
        {'n_estimators': [100, 200, 500]},  # Random Forest Classifier
        {}  # GradientBoostingClassifier
    ]

    return models, models_names, models_hparametes

def create_models():

    models = [
        KNeighborsClassifier(weights='distance'),
        LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced', max_iter=5000),
        SVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced')
    ]

    models_names = ['K-NN', 'Logistic Reg.', 'SVT', 'DT']

    models_hparametes = [
        {'n_neighbors': list(range(1, 10, 2))},  # KNN
        {'penalty': ['l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},
        # SoftmaxReg NB "C" è l'iperprametro di regolarizzazione
        {
            'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001],
            'kernel': ['linear', 'rbf']
        },  # SMV
        {'criterion': ['gini', 'entropy']},  # DT
    ]

    return models, models_names, models_hparametes


def create_classification_ensemble(x_train, y_train):
    title("Classifiers")

    #models, models_names, models_hparametes = create_classification_models_items()
    models, models_names, models_hparametes = create_models()

    best_hparameters = []
    estimators = []

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=cv)
        clf.fit(x_train, y_train)

        best_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf))

        print(model_name)
        print('Accuracy:', clf.best_score_, "\n")

    ensemble_classifier = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
    return ensemble_classifier


def calculate_classification_scores(ensemble, x_train, y_train):
    scores = cross_validate(ensemble, x_train, y_train, cv=cv,
                            scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'))

    accuracy_scores = scores['test_accuracy']
    precision_scores = scores['test_precision_macro']
    recall_scores = scores['test_recall_macro']
    f1_scores = scores['test_f1_macro']

    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)


def print_metrics(accuracy, precision, recall, f1):
    utils.title("[Classificator] Training")

    print("Stacking ensemble - Accuracy: ", accuracy)
    print("Stacking ensemble - Precision: ", precision)
    print("Stacking ensemble - Recall: ", recall)
    print("Stacking ensemble - F1: ", f1)


def print_final_metrics(y_test, y_pred):
    utils.title("[Classification] validation")

    print('Accuracy is ', accuracy_score(y_test, y_pred))
    print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))
