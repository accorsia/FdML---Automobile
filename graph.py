import numpy as np
import matplotlib.pyplot as plt

def plot_correlation(series):
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

def plot_prediction_errors(y_test, y_clf_pred, y_reg_pred):
    # Calcola gli errori di predizione
    clf_errors = y_clf_pred - y_test
    reg_errors = y_reg_pred - y_test

    # Grafico a dispersione per gli errori di predizione
    plt.scatter(y_test, clf_errors, color='blue', label='Classificatori')
    plt.scatter(y_test, reg_errors, color='red', label='Regressori')

    # Aggiungi linee di riferimento per gli errori a zero
    plt.axhline(0, color='black', linestyle='--')

    # Aggiungi etichette agli assi
    plt.xlabel('Valori di ground truth')
    plt.ylabel('Errori di predizione')

    # Aggiungi una legenda
    plt.legend()

    # Mostra il grafico
    plt.show()


def plot_prediction_differences(y_clf_pred, y_reg_pred):
    # Calcola le differenze tra le predizioni
    differences = y_clf_pred - y_reg_pred

    # Calcola il valore assoluto delle differenze
    abs_differences = np.abs(differences)

    # Grafico a barre per le differenze tra le predizioni
    plt.bar(np.arange(len(differences)), abs_differences, color='green')

    # Aggiungi etichette agli assi
    plt.xlabel('Istanze')
    plt.ylabel('Differenze tra predizioni')

    # Mostra il grafico
    plt.show()


def plot_error_by_class(y_test, y_clf_pred, y_reg_pred):
    # Calcola gli errori di classificazione e di regressione per ogni classe
    classes = np.unique(y_test)
    clf_errors = []
    reg_errors = []

    for cls in classes:
        cls_indices = np.where(y_test == cls)
        clf_errors.append(np.sum(y_clf_pred[cls_indices] != cls))
        reg_errors.append(np.sum(np.abs(y_reg_pred[cls_indices] - cls)))

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

    # Mostra il grafico
    plt.show()

if __name__ == "__main__":
    # Carica i valori delle tre variabili y dal file
    data = np.load('y_values.npz')
    y_test = data['y_test']
    y_clf_pred = data['y_clf_pred']
    y_reg_pred = data['y_reg_pred']

    plot_prediction_errors(y_test, y_clf_pred, y_reg_pred)
    plot_prediction_differences(y_clf_pred, y_reg_pred)
    plot_error_by_class(y_test, y_clf_pred, y_reg_pred)