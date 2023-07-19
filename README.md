# Progetto di Machine Learning per l'Analisi dei Dati Automobilistici

Questo progetto utilizza tecniche di machine learning per analizzare e predire dati automobilistici. Il dataset utilizzato contiene informazioni sui veicoli, come il prezzo, le specifiche tecniche, le prestazioni e altro. Lo scopo del progetto è sviluppare modelli di regressione per prevedere il prezzo dei veicoli e modelli di classificazione per classificare i veicoli in diverse categorie.

## Istruzioni per l'esecuzione

1. Installare le dipendenze:
   Assicurarsi di avere tutte le dipendenze necessarie installate. È possibile installarle utilizzando `pip` con il seguente comando:

2. Esecuzione del codice:
Per eseguire il progetto, utilizzare il file `main.py` con il seguente comando:

3. Risultati:
Una volta completata l'esecuzione, i risultati dei modelli di regressione e classificazione verranno visualizzati nel terminale. Inoltre, verranno creati dei file `.npz` contenenti i valori delle metriche di valutazione per ulteriori analisi.

## Struttura del Progetto
* `main.py`: File principale per l'esecuzione del progetto.
* `dataset.py`: Modulo per la gestione del dataset e il pre-processing dei dati.
* `graph.py`: Modulo per la creazione e la visualizzazione di grafici e visualizzazioni.
* `reg.py`: Modulo per la creazione e la valutazione dei modelli di regressione.
* `clf.py`: Modulo per la creazione e la valutazione dei modelli di classificazione.
* `utils.py`: Modulo contenente funzioni di utilità per la visualizzazione e l'analisi dei dati.
* `arguments.py`: Modulo per la lettura degli argomenti da riga di comando.
* `npz`: Directory contenente i file .npz con i valori delle metriche di valutazione.

## Descrizione delle Fasi del Progetto

1. **Caricamento e Pre-processing del Dataset:**
   Il dataset automobilistico è stato pre-processato per rimuovere valori nulli e convertire le variabili categoriche in variabili numeriche. Inoltre, per garantire una rappresentazione equilibrata delle diverse classi, sono state duplicate le righe delle classi sottorappresentate.

2. **Feature Selection e Visualizzazioni dei Dati:**
   È stata eseguita un'analisi delle correlazioni tra le features per selezionare le migliori features per i modelli di regressione e classificazione. Sono stati creati diversi grafici e visualizzazioni per analizzare le relazioni tra le features e le prestazioni dei modelli.

3. **Creazione dei Modelli di Regressione:**
   Sono stati creati diversi modelli di regressione utilizzando algoritmi come Support Vector Regression (SVR), Random Forest Regression e Gradient Boosting Regression. Sono stati eseguiti esperimenti con diversi iperparametri per ogni modello per trovare la migliore configurazione. I modelli sono stati valutati utilizzando la validazione incrociata e sono stati selezionati i migliori modelli per la predizione del prezzo dei veicoli.

4. **Creazione dei Modelli di Classificazione:**
   Sono stati creati modelli di classificazione utilizzando algoritmi come Support Vector Machine (SVM), Random Forest e Gradient Boosting. Sono stati eseguiti esperimenti con diversi iperparametri per ogni modello per trovare la migliore configurazione. I modelli sono stati valutati utilizzando la validazione incrociata e sono stati selezionati i migliori modelli per la classificazione dei veicoli in diverse categorie.

5. **Valutazione dei Modelli:**
   I modelli di regressione e classificazione sono stati valutati utilizzando diverse metriche di valutazione, come Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared e Accuracy. I risultati delle valutazioni sono stati visualizzati tramite grafici e tabelle per una facile interpretazione.

6. **Conclusioni e Prospettive Future:**
   Sono state fornite conclusioni sulle prestazioni dei modelli e su come potrebbero essere migliorati. Sono state anche suggerite prospettive future per ulteriori ricerche e sviluppi del progetto.

---
*Nota: Assicurarsi di avere Python e le librerie necessarie installate per eseguire il progetto.*


