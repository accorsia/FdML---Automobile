import itertools

import matplotlib.pyplot as plt
import numpy as np


def plot_correlation(series):
    plt.title("Plots correlation")

    # Ottenere gli indici e i valori dalla Series
    x = series.index
    y = series.values

    # Creare il grafico a barre
    plt.barh(x, y)

    # Regolare i margini del grafico
    plt.subplots_adjust(left=0.3)

    # Tracciare una linea che congiunge le varie barre
    plt.plot(y, x, linestyle='-', color='red')

    # Aggiungere le etichette agli assi
    plt.xlabel('Valori')
    plt.ylabel('Indici')

    # Aggiungere una legenda
    plt.legend()

    # Mostrare il grafico
    plt.show()


def plot_prediction_differences(y_clf_pred, y_reg_pred):
    plt.title("Scarto tra le predizioni")

    # Calcola il valore assoluto delle differenze tra le predizioni
    abs_differences = np.abs(y_clf_pred - y_reg_pred)

    # Crea il grafico a barre per le differenze tra le predizioni
    plt.bar(np.arange(len(abs_differences)), abs_differences, color='green')

    # Aggiungi etichette agli assi
    plt.xlabel('Istanze')
    plt.ylabel('Differenze tra predizioni')

    # Aggiungi una griglia di sfondo
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Aggiungi una linea orizzontale per il valore medio delle differenze
    mean_difference = np.mean(abs_differences)
    plt.axhline(y=mean_difference, color='red', linestyle='--', label=f'Media: {mean_difference:.2f}')

    # Aggiungi legenda
    plt.legend()

    # Riduci le dimensioni dei margini
    plt.margins(0.05)

    # Applica lo stile 'seaborn-whitegrid' per un aspetto più accattivante
    plt.style.use('seaborn-whitegrid')

    # Mostra il grafico
    plt.show()

    # Ripristina lo stile predefinito
    plt.style.use("default")


def plot_error_by_class(y_test, y_clf_pred, y_reg_pred):
    plt.title("Error by class")

    # Calcola gli errori di classificazione e di regressione per ogni classe
    classes = np.unique(y_test)
    clf_errors = []
    reg_errors = []

    for cls in classes:
        cls_indices = np.where(y_test == cls)
        clf_errors.append(np.sum(y_clf_pred[cls_indices] != cls))
        reg_errors.append(np.sum(y_reg_pred[cls_indices] != cls))

    # Calcola il numero totale di classi
    num_classes = len(classes)

    # Calcola le posizioni delle barre sull'asse x
    bar_positions = np.arange(num_classes)

    # Larghezza delle barre
    bar_width = 0.35

    # Disegna il grafico a barre
    plt.bar(bar_positions, clf_errors, width=bar_width, color='blue', label='Classificazione')
    plt.bar(bar_positions + bar_width, reg_errors, width=bar_width, color='red', label='Regressione')

    # Aggiungi etichette agli assi
    plt.xlabel('Classi di predizione')
    plt.ylabel('Numero di errori')

    # Aggiungi le etichette delle classi sull'asse x
    plt.xticks(bar_positions + bar_width / 2, classes)

    # Aggiungi una legenda
    plt.legend()

    # Aggiungi griglia di sfondo
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Applica lo stile 'seaborn-darkgrid' a tutto 'plt': modifica TEMPORANEA siccome verrà ripristinato lo stile dopo
    plt.style.use('seaborn-darkgrid')

    # Riduci le dimensioni dei margini
    plt.margins(0.05)

    # Mostra il grafico
    plt.show()

    # Ripristina lo stile predefinito per non influenzare gli altri grafici
    plt.style.use("default")


def plot_confusion_matrix(targets, predictions, classes, title, normalize=True, cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.

        Parameters:
            targets (array): Ground truth labels.
            predictions (array): Predicted labels.
            classes (array): Unique class labels.
            normalize (bool, optional): Whether to normalize the confusion matrix. Default is True.
            title (str, optional): Title of the plot. Default is 'Confusion matrix'.
            cmap (colormap, optional): Colormap to use for the plot. Default is plt.cm.Blues.
    """

    # Calcola il numero di classi
    n_classes, = np.unique(targets).shape

    # Inizializza la matrice di confusione
    cm = np.zeros(shape=(n_classes, n_classes), dtype=np.float32)

    # Riempie la matrice di confusione con i conteggi delle predizioni corrette ed errate
    for t, p in zip(targets, predictions):
        cm[int(t), int(p)] += 1

    # Normalizza la matrice di confusione
    if normalize:
        cm /= cm.sum(axis=1)

    # Crea una nuova figura e un nuovo oggetto asse
    fig, ax = plt.subplots(figsize=(8, 6))

    # Disegna la matrice di confusione come un'immagine
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Imposta il titolo del grafico
    ax.set_title(title)

    # Aggiungi una barra dei colori per rappresentare i valori nella matrice
    fig.colorbar(im)

    # Imposta le etichette dell'asse x con i nomi delle classi
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)

    # Imposta le etichette dell'asse y con i nomi delle classi
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Imposta il formato per visualizzare i valori nella matrice
    fmt = '.2f'
    thresh = cm.max() / 2.

    # Aggiungi i valori nella matrice come testo nel grafico
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    # Ridimensiona il grafico in modo ottimale
    plt.tight_layout()

    # Etichetta dell'asse y
    plt.ylabel('True label')

    # Etichetta dell'asse x
    plt.xlabel('Predicted label')

    # Mostra il grafico
    plt.show()


def print_errors_by_class(y_ground_truth, y_regression, y_classification):
    classes = np.unique(y_ground_truth)

    print("Class:\tRegression\tClassification")

    for cls in classes:
        regression_errors = np.sum(y_ground_truth[y_ground_truth == cls] != y_regression[y_ground_truth == cls])
        classification_errors = np.sum(y_ground_truth[y_ground_truth == cls] != y_classification[y_ground_truth == cls])

        print(cls, "\t", regression_errors, "\t", classification_errors)


"""def ground_vs_predict(y_test, y_clf_pred, y_reg_pred):
    # Creazione delle ascisse
    x = np.arange(len(y_test))

    # Creazione del grafico
    plt.plot(x, y_test, label='Valori Effettivi', color='green')
    plt.plot(x, y_clf_pred, label='Previsioni Ensemble di Classificazione', color='blue')
    plt.plot(x, y_reg_pred, label='Previsioni Ensemble di Regressione', color='red')

    # Aggiunta di etichette e titolo
    plt.xlabel('Osservazioni')
    plt.ylabel('Valori')
    plt.title('Confronto tra Previsioni e Valori Effettivi')

    # Aggiunta di legenda
    plt.legend(loc="lower right")

    # Mostrare il grafico
    plt.show()"""


def plot_bar_ground_vs_predict(y_test, y_clf_pred, y_reg_pred):
    # Creazione delle ascisse
    x = np.arange(len(y_test))

    # Creazione del grafico a barre
    width = 0.2
    plt.bar(x - width, y_test, width=width, label='Valori Effettivi', color='green')
    plt.bar(x, y_clf_pred, width=width, label='Previsioni Ensemble di Classificazione', color='blue')
    plt.bar(x + width, y_reg_pred, width=width, label='Previsioni Ensemble di Regressione', color='red')

    # Aggiunta di etichette e titolo
    plt.xlabel('Osservazioni')
    plt.ylabel('Valori')
    plt.title('Confronto tra Previsioni e Valori Effettivi')

    # Aggiunta di legenda
    plt.legend(loc="upper left")

    # Mostrare il grafico
    plt.show()


def plot_reg_metric_changes(mae_train, mse_train, r2_train, mae_test, mse_test, r2_test):
    plt.title('Regression - Metric Comparison')

    # Creazione delle ascisse
    metrics = ['MAE', 'MSE', 'R2']
    train_values = [mae_train, mse_train, r2_train]
    test_values = [mae_test, mse_test, r2_test]

    # Creazione del grafico
    plt.plot(metrics, train_values, 'b-', marker='o', label='Train')
    plt.plot(metrics, test_values, 'r-', marker='o', label='Test')

    # Aggiunta di etichette e titolo
    plt.xlabel('Metrics')
    plt.ylabel('Metric Value')

    # Aggiunta di una legenda
    plt.legend()

    # Mostrare il grafico
    plt.show()


def plot_clf_metrics_changes(accuracy_train, precision_train, recall_train, f1_train,
                             accuracy_test, precision_test, recall_test, f1_test):
    plt.title('Classification - Metric Comparison')

    # Creazione delle ascisse
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    train_values = [accuracy_train, precision_train, recall_train, f1_train]
    test_values = [accuracy_test, precision_test, recall_test, f1_test]

    # Creazione del grafico
    plt.plot(metrics, train_values, 'b-', marker='o', label='Train')
    plt.plot(metrics, test_values, 'r-', marker='o', label='Test')

    # Aggiunta di etichette e titolo
    plt.xlabel('Metrics')
    plt.ylabel('Metric Value')

    # Aggiunta di una legenda
    plt.legend()

    # Mostrare il grafico
    plt.show()


if __name__ == "__main__":
    ###     Data deserialization
    data = np.load('npz/y_values.npz')
    y_test = data['y_test']
    y_clf_pred = data['y_clf_pred']
    y_reg_pred = data['y_reg_pred']

    #   Classification
    data = np.load("npz/clf_test.npz")
    accuracy_test = data["test_accuracy"]
    precision_test = data["test_precision"]
    recall_test = data["test_recall"]
    f1_test = data["test_f1"]

    data = np.load("npz/clf_train.npz")
    accuracy_train = data["train_accuracy"]
    precision_train = data["train_precision"]
    recall_train = data["train_recall"]
    f1_train = data["train_f1"]

    #   Regression
    data = np.load("npz/reg_test.npz")
    mse_test = data["test_mse"]
    mae_test = data["test_mae"]
    r2_test = data["test_r2"]

    data = np.load("npz/reg_train.npz")
    mse_train = data["train_mse"]
    mae_train = data["train_mae"]
    r2_train = data["train_r2"]

    print_errors_by_class(y_test, y_reg_pred, y_clf_pred)  # terminal

    ###     Graph
    plot_reg_metric_changes(mae_train, mse_train, r2_train, mae_test, mse_test, r2_test)
    plot_clf_metrics_changes(accuracy_train, precision_train, recall_train, f1_train,
                             accuracy_test, precision_test, recall_test, f1_test)
    plot_prediction_differences(y_clf_pred, y_reg_pred)
    plot_error_by_class(y_test, y_clf_pred, y_reg_pred)
    plot_bar_ground_vs_predict(y_test, y_clf_pred, y_reg_pred)
    plot_confusion_matrix(targets=y_test, predictions=y_clf_pred, title="Confusion matrix - Classification",
                          classes=np.unique(y_test))
    plot_confusion_matrix(targets=y_test, predictions=y_reg_pred, title="Confusion matrix - Regression",
                          classes=np.unique(y_test))
