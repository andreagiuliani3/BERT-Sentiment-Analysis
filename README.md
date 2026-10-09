# BERT Sentiment Analysis

This project implements a sentiment analysis model based on **BERT** (Bidirectional Encoder Representations from Transformers), specifically fine-tuned on the `bert-base-uncased` variant. The primary goal is to classify text (e.g., movie reviews) into binary categories (e.g., positive/negative).

## 📁 Project Structure

The directory structure of the project is as follows:

```text
Quarto progetto (BERT)/
├── Script/
│   └── BERT.ipynb       # Main notebook containing training and evaluation code
├── README.md            # This file
```

## 📊 Dataset

The dataset used in this project is the **IMDB Movie Ratings Sentiment Analysis**, publicly available on Kaggle. It contains a large collection of movie reviews labeled for sentiment analysis.

🔗 **Dataset Link:** [Kaggle - IMDB Movie Ratings Sentiment Analysis](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis)

*Note: Download the dataset from the link, rename the main file to `movie.csv`, and place it in the project's root directory.*

## 🚀 Requirements and Installation

To run the notebook seamlessly, ensure you have the following Python libraries installed:

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`
- `tabulate`

You can install the dependencies using pip:

```bash
pip install torch torchvision torchaudio transformers pandas numpy scikit-learn matplotlib seaborn nltk tabulate
```

## ⚙️ Data Configuration

The notebook originally included paths related to the Google Colab environment (e.g., Google Drive). 
These have been converted to work optimally on any local machine or GitHub repository.

To test the code:
1. Make sure your dataset file is a CSV file (e.g., `movie.csv`) containing at least the `text` and `label` columns.
2. Open the notebook `Script/BERT.ipynb`.
3. Look for the **Configuration** section in the notebook.
4. Modify the `DATA_PATH` variable by providing the correct path to your local CSV file (by default, it is set to `"../movie.csv"`). 
   - Example if you are running the notebook from the `Script` folder: `DATA_PATH = "../movie.csv"`

## 🧠 Model Workflow

The `BERT.ipynb` file is structured as follows:

1. **Library Import and Configuration:** Setting up the GPU device (if available) and defining hyperparameters.
2. **Data Loading and Preprocessing:** 
   - Data splitting and shuffling.
   - Text cleaning (removing HTML tags, URLs, excessive punctuation, emojis, and stop-words using `nltk`).
3. **Tokenization:** Using `BertTokenizer` to convert sentences into token IDs suitable for BERT, generating attention masks, and enforcing `MAX_LEN`.
4. **DataLoader Setup:** Splitting data into training (80%) and test/validation sets (20%) using PyTorch's `TensorDataset`.
5. **Training (Fine-tuning):** 
   - Initializing `BertForSequenceClassification`.
   - Using the `AdamW` optimizer and HuggingFace's linear learning rate scheduler.
6. **Evaluation and Diagnostics:**
   - Tracking metrics and saving loss trend plots.
   - Evaluation on the test set.
   - Generating a confusion matrix (both normalized and absolute) for proper interpretation.
   - Generating a detailed classification report (Precision, Recall, F1-Score).

## 📊 Results

The notebook is configured to produce logs and plots to measure the model's performance, accurately assessing potential overfitting during the training epochs and printing a summarized textual report using the `tabulate` library.

---
