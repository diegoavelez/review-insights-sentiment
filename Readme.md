# 🧠 Amazon Product Review Sentiment Analysis

This project applies both traditional NLP techniques and modern deep learning to perform **sentiment classification** of Amazon product reviews. It is part of my AI Engineering portfolio to demonstrate end-to-end capabilities in natural language processing, model development, and business insight generation.

---

## 📦 Dataset

* **Source:** [Datafiniti Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
* **Size:** \~28,000 reviews
* **Fields used:**

  * `name` (product name)
  * `primaryCategories` (product category)
  * `reviews.rating` (numeric rating from 1 to 5)
  * `reviews.text` (review content)

A sentiment label was derived using the following logic:

* Rating 1-2: `negative`
* Rating 3: `neutral`
* Rating 4-5: `positive`

---

## 🔍 Exploratory Data Analysis (EDA)

* Sentiment distribution is **heavily skewed** toward positive reviews (\~90%)
* Reviews are short: **median \~17 words**
* Top product categories by volume:

  * Electronics
  * Health & Beauty
  * Toys & Games
* Word analysis revealed:

  * Positive: *great*, *battery*, *tablet*, *good*, *love*
  * Negative: *battery*, *return*, *fail*, *disappointed*, *cheap*

---

## ⚙️ Model 1: TF-IDF + Logistic Regression

### 📐 Pipeline

* `TfidfVectorizer` with grid search over `ngram_range`, `max_df`, `min_df`
* `LogisticRegression` with balanced class weights

### 🧪 Performance

| Metric         | Value  |
| -------------- | ------ |
| Accuracy       | 86.5%  |
| Macro F1 Score | 0.6438 |
| Neutral F1     | 0.3764 |
| Negative F1    | 0.6281 |
| Positive F1    | 0.9271 |

> TF-IDF model provides strong baseline, especially for positive sentiment. Neutral remains difficult.

---

## 🤖 Model 2: DistilBERT (Fine-Tuned Transformer)

### 🛠️ Configuration

* `distilbert-base-uncased` pretrained model
* Fine-tuned using HuggingFace `Trainer`
* 3-class classification head
* Tokenized with `DistilBertTokenizerFast`

### 📈 Performance on Test Set

| Metric         | Value      |
| -------------- | ---------- |
| Accuracy       | **94.3%**  |
| Macro F1 Score | **0.7214** |
| Neutral F1     | 0.4612     |
| Negative F1    | 0.7291     |
| Positive F1    | 0.9740     |

> DistilBERT significantly improves classification, especially for minority classes. Shows superior generalization.

---

## 🧪 Inference Example

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("./models/bert_sentiment")
tokenizer = DistilBertTokenizerFast.from_pretrained("./models/bert_sentiment")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()
    return pred  # Use inverse label encoder to return class name

predict_sentiment("This product is amazing and works perfectly!")
```

---

## 💾 Model Export

The fine-tuned DistilBERT model is saved under:

```
./models/bert_sentiment/
```

Includes:

* `pytorch_model.bin`
* `config.json`
* `tokenizer_config.json`, `vocab.txt`

This allows reloading for inference or deployment with HuggingFace APIs.

---

## 📌 Key Skills Demonstrated

* NLP preprocessing and text normalization
* Feature engineering with TF-IDF
* Model optimization via `GridSearchCV`
* Deep learning with HuggingFace Transformers
* Metric interpretation (macro F1, class imbalance)
* Exporting production-ready models

---

## 📍 Project Structure

```
📁 data/
    raw_dataset.csv
📁 notebooks/
    review_sentiment_classification.ipynb
📁 models/
    bert_sentiment/
📁 reports/
    summary.pdf
    confusion_matrix_distilbert.png
📄 requirements.txt
📄 README.md
📄 CASE_STUDY.md
```

---

## 🧠 Author

**Diego Alejandro Vélez Martínez**
Data Scientist & AI Engineer
[GitHub](https://github.com/diegoavelez) • [LinkedIn](https://www.linkedin.com/in/diegoavelezm/)

---

## 📄 License

This project is licensed under the MIT License.
