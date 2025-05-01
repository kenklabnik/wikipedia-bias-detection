# Wikipedia Bias Detection — MayCodePudding 🧠📰

This project analyzes text from Wikipedia and news headlines to detect political or ideological bias using machine learning. It uses a combination of labeled datasets, a bias word lexicon, and TF-IDF-based classification to build a supervised bias detector. It also supports scraping live Wikipedia articles and generating sentence-level bias scores.

## 🔍 Project Goal

Wikipedia is written and edited by the public, which means bias can unintentionally slip through. This project aims to:

- Train a supervised machine learning model on real-world labeled sentences
- Predict sentence-level bias in any given Wikipedia article
- Provide a measurable bias score based on sentence predictions
- Enable batch scraping and analysis of politically sensitive topics

## 🧱 Key Features

- Uses TF-IDF vectorization + logistic regression for classification
- Handles text preprocessing, categorical encoding, and feature scaling in one pipeline
- Supports sentence-by-sentence bias detection with adjustable thresholds
- Automatically scrapes and evaluates Wikipedia articles by topic
- Tracks sentence bias probability and keywords using a custom lexicon


## 📊 Datasets Used

- **Final Labels SG1 / SG2 / MBIC** — Real-world political sentences labeled for bias
- **Bias Word Lexicon** — List of potentially biased terms
- **Headline Sets** — Biased and neutral news headlines used for feature support

## 🧪 Model Performance

- **Accuracy**: ~89%
- **F1-Score**: ~0.89
- **ROC AUC**: 0.95  
- Best results achieved using `max_features=20000`, `ngram_range=(1,3)`, and `C=1.0`

## 🌐 Wikipedia Integration

You can enter any Wikipedia article title, and the system will:

- Scrape its content
- Break it into sentences
- Predict bias per sentence
- Return an overall bias score

## 🚀 Try It Out

To run the prediction on a Wikipedia article:

```python
results = predict_bias_from_article("Donald Trump", pipeline)
print(f"Bias Score: {results['bias_score']} ({results['biased_sentences']} of {results['total_sentences']} sentences)")
