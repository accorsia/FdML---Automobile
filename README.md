# Progetto di regressione

Questo progetto è un'applicazione di regressione utilizzando modelli di machine learning per prevedere la variabile target "symboling" in un dataset di automotive.

## Descrizione del progetto

Il progetto utilizza un dataset di automotive per addestrare e valutare modelli di regressione. Il dataset viene preelaborato per gestire i valori mancanti, i tipi di dati e le variabili categoriche. Successivamente, viene applicata la selezione delle feature basata sulla matrice di correlazione per estrarre le migliori N feature.

## Struttura del progetto

Il progetto è suddiviso nei seguenti file:

- `main.py`: Contiene il codice principale per l'addestramento, la valutazione e la visualizzazione dei risultati dei modelli di regressione.
- `dataset.py`: Contiene funzioni per il recupero e la preelaborazione del dataset.
- `utils.py`: Contiene funzioni di utilità per la visualizzazione delle informazioni sul dataset e dei risultati.

## Istruzioni per l'utilizzo

1. Assicurarsi di avere Python 3 installato.
2. Clonare il repository sul proprio computer.
3. Installare le dipendenze eseguendo il comando `pip install -r requirements.txt`.
4. Eseguire il comando `python main.py` per avviare il progetto.

## Configurazione del progetto

Il progetto offre alcune opzioni di configurazione attraverso l'utilizzo di argomenti da riga di comando. Di seguito sono elencati gli argomenti disponibili:

- `--dataset_path_train`: Percorso del file contenente il dataset di addestramento (valore predefinito: "dataset/imports-85.data").
- `--cv`: Numero di fold per la cross-validation (valore predefinito: 2).
- `--nan_tolerance`: Soglia di tolleranza per la percentuale di valori mancanti dopo la quale la colonna verrà eliminata (valore predefinito: 0.5).
- `--train_test_ratio`: Rapporto tra i dati di addestramento e test rispetto all'intero dataset (valore predefinito: 0.2).
- `--n_features`: Numero di migliori feature da estrarre dalla matrice di correlazione (valore predefinito: 13).

È possibile modificare questi argomenti per adattare il progetto alle proprie esigenze.

## Risultati

I risultati dell'addestramento e della valutazione dei modelli vengono visualizzati nella console e vengono generati grafici per confrontare i valori previsti con i valori effettivi.

## Dipendenze

Il progetto richiede le seguenti dipendenze Python:

- numpy
- pandas
- matplotlib
- scikit-learn

## Contributi

Sono benvenuti contributi e suggerimenti per migliorare il progetto. Se desideri contribuire, puoi creare una nuova branch, aprire una pull request e descrivere le modifiche apportate.

## Licenza

Il progetto è distribuito con la licenza MIT