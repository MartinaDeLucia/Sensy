![Sensy logo](img/img.png "SENSY")
# README - Pipeline di Clustering e Classificazione per Domande Sensitive

## Descrizione del Progetto

Questo progetto fornisce una pipeline **end-to-end** per individuare e classificare domande “**sensitive**” (ad esempio contenenti contenuti violenti, discriminatori o simili) o “non-sensitive”. I principali step della pipeline sono:

1. **Clustering** delle domande in base alle loro embedding (utilizzando **Sentence-BERT** e **HDBSCAN**).
2. **Generazione** di dataset JSON di training (domande sensitive / non-sensitive) con gestione del “noise”.
3. **Addestramento** di un modello di classificazione (Random Forest).
4. **Valutazione** delle performance (accuracy, precision, recall, F1-score).
5. **Predizione** su singole domande tramite un modello addestrato.

## Struttura del Repository

La struttura generale è la seguente:
```
.
├── clustering
│   ├── generate_sensitive_from_clusters.py
│   └── question_clustering.py
├── data
│   ├── question_train.json
│   ├── question_test.json
│   └── normalized_chatgpt_questions.json
├── models
│   ├── evaluate_model.py
│   └── train_model.py
├── preprocessing
│   ├── clean_data.py
│   ├── dataset_report.py
│   └── feature_extraction.py
├── samples
│   ├── models
│   ├── report
│   └── results
├── common_functions.py
├── extract_single.py
├── predict_sensitive.py
├── main.py
└── README.md  <– questo file
```

### Breve Descrizione dei File Principali

1. **`clustering/question_clustering.py`**  
   - Carica il dataset (domande sensitive e non-sensitive).
   - Estrae gli embedding (Sentence-BERT), riduce la dimensionalità con PCA.
   - Esegue il clustering con **HDBSCAN** usando la **cosine distance**.
   - Organizza i risultati e salva i cluster e le statistiche (file JSON).

2. **`clustering/generate_sensitive_from_clusters.py`**  
   - Genera file JSON combinando i cluster **sensitive** e **non-sensitive**, tenendo conto di possibili punti di “noise” (etichettati `-1`).
   - Produce in output tre file:
     - `question_train_sensitive_clustered_output.json`
     - `question_train_nonsensitive_clustered_output.json`
     - `total_clusters_question_train.json` (unione dei due precedenti).

3. **`common_functions.py`**  
   - Funzioni comuni, tra cui:
     - *POS tagging* (`shared_count_pos_tags`)  
     - *Conteggio parole “sensibili”* (`shared_count_sensitive_words`)  
     - *Sentiment analysis* (usando `nltk.sentiment.SentimentIntensityAnalyzer`)  
     - *Generazione embedding BERT* (singolo e batch) caricando **BertTokenizer** e **BertModel** una sola volta.  
   - Sfrutta il device migliore disponibile (CPU/GPU/MPS) per velocizzare i calcoli.

4. **`preprocessing/clean_data.py`**  
   - Pulisce i dataset JSON (rimozione duplicati, eventuale drop di colonne non necessarie, tokenizzazione).

5. **`preprocessing/feature_extraction.py`**  
   - Estrae le feature (numero di verbi, nomi, aggettivi, parole sensibili, sentiment…) + embedding BERT in **batch** per un intero dataset (DataFrame).

6. **`preprocessing/dataset_report.py`**  
   - Genera un report (CSV) con statistiche di base (duplicati, conteggio domande sensitive/non-sensitive, ecc.) su uno o più dataset.

7. **`extract_single.py`**  
   - Fornisce funzioni per estrarre feature da **una singola domanda** (POS, parole sensibili, embedding BERT [CLS], sentiment).

8. **`predict_sensitive.py`**  
   - Script interattivo per effettuare una **predizione** su una singola domanda, caricando il modello addestrato (file `.pkl`).

9. **`models/train_model.py`**  
   - Addestra un modello di classificazione (qui **Random Forest**) con possibilità di definire se eseguire uno split train/validation interno o usare tutti i dati.

10. **`models/evaluate_model.py`**  
    - Fornisce la funzione `evaluate_model` per valutare il modello (accuracy, precision, recall, F1-score), restituendo un *classification report*.

11. **`main.py`**  
    - Coordina l’intero flusso di training:  
      1. Caricamento/pulizia del dataset principale (`total_clusters_question_train.json`), del test set, e di un dataset aggiuntivo (opzionale).  
      2. Bilanciamento dei dati (in base alle percentuali `ADDITIONAL_TRAIN_RATIOS`).  
      3. Estrazione feature (`preprocessing/feature_extraction.py`).  
      4. Addestramento del modello (`train_model`).  
      5. Valutazione (`evaluate_model`) e salvataggio dei risultati (modelli, report, CSV finale).

---

## Requisiti

- **Python 3.8+** (consigliato 3.9 o 3.10).
- Librerie: `numpy`, `pandas`, `scikit-learn`, `nltk`, `hdbscan`, `transformers`, `sentence-transformers`, `torch`, `pickle`, `json`, ecc.
- Installa con
    ```
    pip install numpy pandas scikit-learn nltk hdbscan torch transformers sentence-transformers
  ```
- Ricordati di scaricare i corpora NLTK necessari:
    ```
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
  ```
  
---

## Come Eseguire Passo-Passo
### Preparazione dei Dati

Assicurati di avere i file nella cartella data/:
- question_train.json: dataset di training iniziale (domande con sensitive? = 0 o 1).
- question_test.json: dataset di test.
- normalized_chatgpt_questions.json: (opzionale) dataset aggiuntivo da integrare in fase di training.
### Clustering delle domande

1. Vai nella cartella clustering/ (o resta nella root ed esegui l’appropriato percorso relativo).
2.	Avvia lo script question_clustering.py: python clustering/question_clustering.py

Questo genera i file sensitive_clusters.json, non_sensitive_clusters.json e le relative statistiche (sensitive_statistics.json, non_sensitive_statistics.json) all’interno di clustering/export/.

### Generazione del Dataset di Training (Sensitive e Non-Sensitive)
Una volta ottenuti i file di cluster, esegui:
```
python clustering/generate_sensitive_from_clusters.py
```

Genererà:
- question_train_sensitive_clustered_output.json (domande sensitive)
- question_train_nonsensitive_clustered_output.json (domande non-sensitive)
- total_clusters_question_train.json (unione dei due precedenti)

### Addestramento e Valutazione del Modello
Esegui il file main.py:
```
python main.py
```
- Passaggi (automazioni):
  1.	Caricamento/pulizia dei file JSON (train, test, additional).
  2.	Bilanciamento del dataset (in base a ADDITIONAL_TRAIN_RATIOS).
  3.	Estrazione feature: feature sintattiche/semantiche + embedding BERT in batch.
  4.	Addestramento modello (Random Forest) su train set combinato.
  5.	Valutazione sul test set, salvataggio report e salvataggio del modello .pkl.
- In output, vengono creati (nelle rispettive cartelle di samples/):
  - Modelli: samples/models/model_square_testset_ratio_*.pkl
  - Report: samples/report/report_squaretestset_*.json
  - Risultati: samples/results/training_results.csv con l’accuracy ).

### Predizione su Singola Domanda
Dopo aver addestrato il modello, puoi testare la predizione su una domanda in inglese:
```
python predict_sensitive.py
```

Lo script:
1.	Carica il modello di default (es. model_square_testset_ratio_1.0.pkl) da samples/models/.
2.	Chiede in input la domanda.
3.	Estrae le feature (extract_single.py) e predice con model.predict().
4.	Restituisce a schermo se la domanda è considerata SENSITIVE o NON-sensitive.

### File di Output Principali
```
1. Clustering:
    - clustering/export/sensitive_clusters.json
    -	clustering/export/non_sensitive_clusters.json
    -	clustering/export/sensitive_statistics.json
    -	clustering/export/non_sensitive_statistics.json
    
2. Dataset di Training (generati da generate_sensitive_from_clusters.py):
    - clustering/export/question_train_sensitive_clustered_output.json
    -	clustering/export/question_train_nonsensitive_clustered_output.json
    -	clustering/export/total_clusters_question_train.json
    
3. Modelli Addestrati (cartella samples/models/):
    -	model_square_testset_ratio_*.pkl
    
4. Report di Valutazione (cartella samples/report/):
    -	report_squaretestset_*.json
    
5. Risultati Finali (cartella samples/results/):
    -	training_results.csv
```

## Autori
- [Gregorio Garofalo] – Sviluppo e integrazione dell’intero progetto.
- [Martina De Lucia] – Sviluppo e integrazione dell’intero progetto.
- [Alessandra Raia] – Sviluppo e integrazione dell’intero progetto.
