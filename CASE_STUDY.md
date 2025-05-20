
# 🧠 Case Study: Review Insights – Sentiment Classification

## 📌 Context
As part of my AI Engineering portfolio, I developed a sentiment classification model using Amazon product reviews. The goal was to explore both traditional NLP methods and fine-tuned transformer models to extract business-relevant insights and demonstrate deployable AI skills.

## 🎯 Objective
Classify customer reviews into three sentiment classes — positive, neutral, and negative — by:
- Applying baseline models using TF-IDF + Logistic Regression
- Fine-tuning a transformer-based model (DistilBERT)
- Comparing performance metrics and visualizing results
- Generating business insights from real user feedback

## 🗂 Dataset
- **Source:** Datafiniti Amazon Reviews (via Kaggle)
- **Volume:** ~28,000 reviews
- **Fields used:** Product name, category, rating, review text
- **Labeling logic:**
  - Ratings 1–2 → Negative
  - Rating 3 → Neutral
  - Ratings 4–5 → Positive

## 🔍 Exploratory Analysis
- Class imbalance: ~89% of reviews are positive
- Short-form content: Median review length = 17 words
- Frequent words (by sentiment):
  - Positive: great, battery, love, tablet
  - Negative: return, fail, disappointed
  - Neutral: okay, buy, app
- Top categories: Electronics, Health & Beauty, Toys

## ⚙️ Models & Performance

### 1. TF-IDF + Logistic Regression

| Metric     | Score   |
|------------|---------|
| Accuracy   | 86.5%   |
| Macro F1   | 0.64    |
| Neutral F1 | 0.38    |
| Negative F1| 0.62    |
| Positive F1| 0.93    |

✅ Simple and interpretable, but weak on neutral sentiment.

### 2. Fine-tuned DistilBERT

| Metric     | Score   |
|------------|---------|
| Accuracy   | 94.3%   |
| Macro F1   | 0.72    |
| Neutral F1 | 0.46    |
| Negative F1| 0.73    |
| Positive F1| 0.97    |

✅ Superior generalization and strong performance on minority classes.

## 📊 Confusion Matrix – DistilBERT

```
                 Predicted
             Neg   Neu   Pos
Actual Neg   214    59    36
       Neu    79    92    66
       Pos    15    65  4988
```

## ✅ Strategic Insights
- DistilBERT significantly improved sentiment classification across all metrics
- Positive reviews dominate volume, but critical outliers (negative/neutral) are now better detected
- The model can be deployed for real-time review monitoring, product feedback analysis, or customer satisfaction tracking

## 💾 Deployment-Ready Model
- Tokenizer + fine-tuned weights saved to `/models/bert_sentiment/`
- Reloadable via HuggingFace’s Transformers for real-time inference
- Includes inference wrapper function for product teams

## 👨‍💻 Author
**Diego Alejandro Vélez Martínez**  
AI Engineer in Training | Senior Project Manager  
[🔗 GitHub](https://github.com/diegoavelez) · [🔗 LinkedIn](https://linkedin.com/in/diegoavelezm)
