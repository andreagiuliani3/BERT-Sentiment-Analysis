# BERT Sentiment Analysis

Questo progetto implementa un modello di analisi del sentiment basato su **BERT** (Bidirectional Encoder Representations from Transformers), specificamente pre-addestrato sulla variante `bert-base-uncased`. L'obiettivo principale è classificare i testi (es. recensioni di film) in categorie binarie (es. positivo/negativo).

## 📁 Struttura del progetto

La struttura delle directory del progetto è la seguente:

```text
Quarto progetto (BERT)/
├── Script/
│   └── BERT.ipynb       # Notebook principale con il codice per il training e la valutazione
├── Immagini/            # Directory contenente eventuali grafici e immagini generati
├── Esercitazioni/       # File e script di supporto/esercitazione
├── README.md            # Questo file
└── movie.csv            # Dataset (da inserire se non presente)
```


## 📊 Dataset

Il dataset utilizzato in questo progetto è l'**IMDB Movie Ratings Sentiment Analysis**, disponibile pubblicamente su Kaggle. Contiene un'ampia raccolta di recensioni cinematografiche etichettate per l'analisi del sentiment.

🔗 **Link al dataset:** [Kaggle - IMDB Movie Ratings Sentiment Analysis](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis)

*Nota: Scarica il dataset dal link e rinomina il file principale in movie.csv, posizionandolo nella directory principale del progetto.*

## 🚀 Requisiti e Installazione

Per eseguire il notebook senza problemi, assicurati di avere installate le seguenti librerie Python:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`
- `tabulate`

Puoi installare le dipendenze usando pip:

```bash
pip install torch torchvision torchaudio transformers pandas numpy scikit-learn matplotlib seaborn nltk tabulate
```

## ⚙️ Configurazione dei dati

Il notebook originariamente includeva percorsi relativi all'ambiente Google Colab (es. Google Drive). 
Questi sono stati convertiti per funzionare in modo ottimale su qualsiasi macchina locale o repository GitHub.

Per testare il codice:
1. Assicurati che il file del dataset sia un file CSV (es. `movie.csv`) contenente almeno le colonne `text` e `label`.
2. Apri il notebook `Script/BERT.ipynb`.
3. Cerca la sezione **Configuration** nel notebook.
4. Modifica la variabile `DATA_PATH` inserendo il percorso corretto verso il tuo file CSV locale (di default è impostato come `"../movie.csv"`). 
   - Esempio se esegui il notebook dalla cartella `Script`: `DATA_PATH = "../movie.csv"`

## 🧠 Flusso di lavoro del modello

Il file `BERT.ipynb` è stato strutturato e commentato in modo professionale per coprire tutti gli step essenziali:

1. **Importazione delle Librerie e Configurazione:** Setup del device GPU se disponibile e definizione di iperparametri.
2. **Caricamento e Preprocessing dei dati:** 
   - Suddivisione e rimescolamento (shuffle) dei dati.
   - Pulizia del testo da tag HTML, URL, punteggiatura in eccesso, emoji e stop-words tramite `nltk`.
3. **Tokenizzazione:** Utilizzo di `BertTokenizer` per convertire le frasi in ID token utilizzabili da BERT, compresa la generazione delle attention masks e la limitazione alla `MAX_LEN`.
4. **Setup del DataLoader:** Divisione in training (80%) e test/validation set (20%) tramite PyTorch `TensorDataset`.
5. **Addestramento (Fine-tuning):** 
   - Inizializzazione di `BertForSequenceClassification`.
   - Utilizzo dell'ottimizzatore `AdamW` e dello scheduler del learning rate lineare di HuggingFace.
6. **Valutazione e Diagnostica:**
   - Tracciamento delle metriche e salvataggio dei grafici di andamento della loss.
   - Valutazione sul test set.
   - Matrice di confusione (sia normalizzata che assoluta) per una corretta interpretazione.
   - Generazione di un classification report dettagliato (Precision, Recall, F1-Score).

## 📊 Risultati

Il notebook è configurato per produrre log e grafici professionali per misurare la performance del modello, calcolando accuratamente eventuali casi di overfitting durante il ciclo delle ereche e stampando risultati testuali riassuntivi grazie alla libreria `tabulate`.

---

**Nota:** Se si utilizza GitHub o altri sistemi di versioning, ricordarsi di aggiungere il file del dataset (`.csv`) al file `.gitignore` per evitare di committare grandi moli di dati.

