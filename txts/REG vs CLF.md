# Regression vs Classification
## Bagging (Bootstrap Aggregating)

Si addestrano diversi modelli indipendenti su <u>sottoinsiemi casuali del dataset di addestramento</u>
utilizzando il <u>campionamento con sostituzione (bootstrap)</u>. 
I risultati dei modelli vengono quindi combinati attraverso 
- una media (nel caso della regressione)
- una votazione (nel caso della classificazione)

**Vantaggi:**
- Riduce la varianza del modello riducendo l'overfitting.
- Funziona bene con modelli instabili o sensibili al rumore.
- Può gestire dataset di grandi dimensioni e complessi.

**Svantaggi:**
- Non migliora la bontà del modello rispetto ai singoli modelli di base.
- Può essere computazionalmente intensivo a causa della necessità di addestrare e combinare diversi modelli.

---

## Boosting

Il Boosting è un metodo di ensemble in cui si addestrano iterativamente dei modelli di base in sequenza, 
dove <u>ogni modello successivo si concentra sui campioni che sono stati classificati in modo errato dai modelli precedenti.</u> 
I risultati dei modelli vengono combinati attraverso un processo di pesatura.

**Vantaggi:**
- Migliora la bontà del modello rispetto ai singoli modelli di base.
- Funziona bene con dataset complessi e rumorosi.
- Può dare risultati eccellenti con modelli di base deboli.

**Svantaggi:**
- Può essere sensibile al rumore e agli outliers nel dataset.
- Potenziale rischio di overfitting se il numero di iterazioni è troppo elevato.

---

## Stacking

Lo Stacking è un metodo di ensemble che combina diversi modelli di base attraverso 
un <u>modello meta</u> che apprende come combinare i risultati dei modelli di base. 
I modelli di base possono essere di diversi tipi e la combinazione dei risultati può essere fatta attraverso 
una regressione lineare, una regressione logistica o altri metodi.

**Vantaggi:**
- Combina le forze dei modelli di base per ottenere prestazioni migliori.
- Può catturare relazioni complesse nel dataset utilizzando modelli diversi.
- Flessibilità nel progettare l'architettura dell'ensemble.

**Svantaggi:**
- Maggior complessità rispetto agli altri metodi di ensemble.
- Richiede più tempo e risorse computazionali per l'addestramento e la valutazione.
