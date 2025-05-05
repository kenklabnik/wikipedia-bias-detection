# Wikipedia Bias Detection — MayCodePudding

This project investigates ideological bias in Wikipedia articles using expert-labeled training data and machine learning. It is built on a refined pipeline ("Randy’s Update") that enhances the original model with deeper text features, more consistent predictions, and interpretability tools. The system uses a custom lexicon of bias-related words and merges headline context into the input to strengthen signal detection.

---

## 📌 Project Goal

Wikipedia is written by the public and can reflect unintentional bias. This project aims to:

- Train a supervised ML model to detect political or ideological bias at the sentence level
- Generate article-wide bias scores based on aggregated sentence predictions
- Use labeled news datasets and lexical features for improved precision
- Scrape and analyze live Wikipedia articles for bias quantification

---

## 🚀 Key Features

- **Enhanced model pipeline** using `TfidfVectorizer`, `StandardScaler`, and `LogisticRegression`
- **Merged input features** from sentence and headline (`combined_text`)
- **Lexicon-based feature engineering** (`lexicon_match_count`)
- Sentence-level bias prediction with output probabilities
- Full Wikipedia scraping and scoring workflow

---

## 📂 Datasets Used

- `final_labels_SG2.csv` — Expert-labeled dataset (SG2 from BABE corpus)
- `bias_word_lexicon.xlsx` — List of bias-indicative terms
- `news_headlines_usa_biased.csv` & `news_headlines_usa_neutral.csv` — Merged with SG2 for headline context

---

## 📊 Model Performance (Final Pipeline)

- **Accuracy**: ~89%
- **ROC AUC**: ~0.95
- Lexicon and headline-enhanced features improve both generalization and interpretability

---

## 🔍 Try It Out

Use the function below to test the model on a Wikipedia article:

```python
results = predict_bias_from_article("Donald Trump", pipeline)
print(f"Bias Score: {results['bias_score']} ({results['biased_sentences']} of {results['total_sentences']} sentences)")
```

---

## 🖥️ How to Run This Repository Locally

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/MayCodePudding-Randy.git
cd MayCodePudding-Randy
```

2. **Create and activate a virtual environment**  
```bash
python3 -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Launch the notebook**  
```bash
jupyter notebook
```

5. **Open and run** `notebooks/randy_bias_analysis.ipynb`

---

## 🛠️ Requirements

- Python 3.8+
- pip
- Jupyter Notebook

---

Let me know if you'd like help updating the `requirements.txt` file next or auto-generating a minimal version based on the final notebook.