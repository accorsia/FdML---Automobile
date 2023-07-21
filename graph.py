import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sea

import arguments


def plot_class_instances(values, title):
    # Etichette delle classi, da -3 a +3
    labels = np.arange(-3, 4)

    # Calcola la media dei valori
    mean_value = np.mean(values)

    # Creazione del grafico a barre
    plt.bar(labels, values, color='skyblue', edgecolor='black')

    # Aggiungi una riga per indicare la media dei valori
    plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Media: {mean_value:.2f}')

    # Aggiungi etichette agli assi e un titolo
    plt.xlabel('Classi')
    plt.ylabel('Numero di istanze')
    plt.title('Numero di istanze per classe: ' + title)

    # Aggiungi una legenda
    plt.legend()

    # Mostra il grafico
    plt.show()


def plot_features_correlation(series):
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

    # Aggiungere una linea orizzontale nera alla decima barra
    plt.axhline(y=9.5, color='black', linestyle='--',
                label=str(arguments.read_args().n_features) + ' Best features split')

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


def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))  # Adjust the figure size as per your preference
    sea.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".1f")
    plt.tight_layout()  # Adjust layout to prevent labels from getting cut off
    plt.show()


def plot_ensembles_common_accuracy(perc_clf, perc_reg):
    ind = [1, 2]  # Bar indexes
    accuracies = [perc_clf, perc_reg]  # Bar values
    labels = ['Classification', 'Regression']  # Labels
    colors = ['#1f77b4', '#2ca02c']  # Bar colors

    # Set seaborn style only for the bar chart
    with sea.axes_style('whitegrid', {'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.4'}):
        plt.bar(ind, accuracies, color=colors, label=labels)  # Create the bar chart
        plt.title("Validation test Accuracy comparison")

    # Add labels to axes
    plt.xlabel('Ensembles', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')

    plt.ylim(0, 1)  # Set y-axis limits to reach up to 1

    # Add labels to the bars with accuracy values
    for i, acc in zip(ind, accuracies):
        plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    plt.xticks(ind, ['Classification', 'Regression'])
    plt.legend(loc="lower right")
    plt.show()

    # Restore the default style
    sea.set_style('ticks')


def plot_r2(r2_dict):
    x_regressors = [x.replace("Regressor", "") for x in list(r2_dict.keys())]
    r2_values = list(r2_dict.values())

    mean_r2 = np.mean(r2_values)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots()

    ax.bar(x_regressors, r2_values, color=colors)
    ax.set_ylim(0, max(r2_values) + 0.15)

    ax.axhline(y=mean_r2, color='red', linestyle='dashed', label=f'Media ({mean_r2:.2f})')

    ax.set_xlabel('Modelli', fontweight='bold')
    ax.set_ylabel('Valore R2', fontweight='bold')

    for i, v in enumerate(r2_values):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', alpha=0.8)

    ax.legend(loc='best')
    ax.set_title('Confronto R2 dei Modelli', fontweight='bold')
    plt.tight_layout()
    plt.show()



def plot_accuracy(accuracy_dict):
    # Estrai i nomi dei modelli e i valori di accuratezza dal dizionario
    x_classifiers = [x.replace("Classifier", "") for x in
                     list(accuracy_dict.keys())]  # Rimuovi 'Classifier' dalle etichette x
    accuracy_values = list(accuracy_dict.values())

    # Calcola la media dei valori di accuratezza
    mean_accuracy = np.mean(accuracy_values)

    # Colori per il grafico
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Crea la figura e l'asse
    fig, ax = plt.subplots()

    # Aggiungi l'istogramma
    ax.bar(x_classifiers, accuracy_values, color=colors)
    ax.set_ylim(0, max(accuracy_values) + 0.15)
    # Aggiungi la linea media
    ax.axhline(y=mean_accuracy, color='red', linestyle='dashed', label=f'Media ({mean_accuracy:.2f})')

    # Aggiungi le etichette dell'asse x e y con stile grassetto
    ax.set_xlabel('Modelli', fontweight='bold')
    ax.set_ylabel('Accuratezza', fontweight='bold')
    ax.set_title('Confronto Accuratezza dei Modelli', fontweight='bold')

    # Aggiungi le etichette sopra le barre con maggiore trasparenza
    for i, v in enumerate(accuracy_values):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', alpha=0.8)

    # Aggiungi una legenda
    ax.legend(loc="best")

    # Mostra il grafico
    plt.tight_layout()
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

    #   Accuracy
    data = np.load("npz/final_accuracy.npz")
    reg_acc = data["reg_acc"]
    reg_clf = data["clf_acc"]

    print_errors_by_class(y_test, y_reg_pred, y_clf_pred)  # terminal

    #   regression models - R2
    data = np.load("npz/reg_models_r2.npz", allow_pickle=True)
    r2_dict = data["dict"].item()

    #   regression models - R2
    data = np.load("npz/clf_models_accuracy.npz", allow_pickle=True)
    clf_dict = data["dict"].item()

    ####################################################################################################################

    ###     Graph
    plot_r2(r2_dict)
    plot_accuracy(clf_dict)

    plot_reg_metric_changes(mae_train, mse_train, r2_train, mae_test, mse_test, r2_test)
    plot_clf_metrics_changes(accuracy_train, precision_train, recall_train, f1_train,
                             accuracy_test, precision_test, recall_test, f1_test)
    plot_prediction_differences(y_clf_pred, y_reg_pred)
    plot_error_by_class(y_test, y_clf_pred, y_reg_pred)
    plot_bar_ground_vs_predict(y_test, y_clf_pred, y_reg_pred)
    plot_ensembles_common_accuracy(reg_clf, reg_acc)
    plot_confusion_matrix(targets=y_test, predictions=y_clf_pred, title="Confusion matrix - Classification",
                          classes=np.unique(y_test))
    plot_confusion_matrix(targets=y_test, predictions=y_reg_pred, title="Confusion matrix - Regression",
                          classes=np.unique(y_test))
