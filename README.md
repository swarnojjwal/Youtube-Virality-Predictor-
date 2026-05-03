# 🎬 YouTube Video Virality Predictor

A machine learning-based binary classification system that predicts whether a YouTube video will go viral, trained on early trending-page statistics from 161,000+ records across four countries.

> **Accuracy: ~89% | Model: Random Forest | Dataset: YouTube Trending (US, CA, GB, FR)**

---

## 📌 Project Overview

YouTube receives over 500 hours of uploaded content every minute, yet only a fraction of videos achieve mass virality. This project answers a key question:

> *Given a YouTube video's statistics at the time it first appears on the trending page, can we predict whether it will eventually achieve above-median peak views?*

This is a real-world **early-warning system** — the model observes initial signals and predicts future outcomes, rather than merely classifying already-viral content.

---

## 📁 Repository Structure
├── mlvirality_FINAL_1.ipynb   
└── README.md
---

## 📊 Dataset

- **Source**: YouTube Trending Videos Dataset (publicly available)
- **Size**: 161,470 rows before preprocessing → ~55,000 unique videos after deduplication
- **Countries**: United States, Canada, Great Britain, France
- **Period**: Pre-November 2021 (dislike counts publicly available)

### Features Engineered

| Feature | Description |
|---|---|
| `views`, `likes`, `dislikes`, `comment_count` | Raw engagement metrics at time of trending snapshot |
| `like_ratio` | `likes / (likes + dislikes + 1)` — audience positivity |
| `comment_ratio` | `comment_count / (likes + 1)` — discussion intensity |
| `dislike_ratio` | `dislikes / (likes + dislikes + 1)` — negative reception |
| `title_length` | Number of words in the video title |
| `tags_count` | Number of tags used |
| `hour`, `day_num` | Publish hour and day of week |
| `category_id` | YouTube category (10 = Music, 24 = Entertainment, etc.) |
| `country_enc` | Label-encoded publish country |

### Target Variable

A video is labelled **viral (1)** if its `peak_views` exceed the **median peak views of the training set**, and **non-viral (0)** otherwise. This produces a naturally balanced 50/50 class split — no oversampling required.

---

## 🔍 Key EDA Findings

- Views distribution is **heavily right-skewed**; log-transformed distribution is approximately normal
- **Likes, comment_count, and like_ratio** are the strongest correlates with peak views
- Videos published between **3 AM–6 AM UTC** (early morning) show a spike in average views
- **Friday** publishes yield the highest average views across the week
- **Music (Category 10)** dominates average views by a significant margin over all other categories
- **GB-origin videos** achieve the highest average peak views among the four countries

---

## ⚙️ Methodology

### Preprocessing
1. Drop null values
2. Extract `peak_views` per video (maximum views across all trending appearances)
3. Keep only the **earliest trending snapshot** per video (to simulate early-prediction)
4. Parse dates, encode categorical columns, engineer ratio features
5. Cast boolean flags to integers

### Train/Test Split
- 80% training / 20% test
- Fixed `random_state=42` for reproducibility

### Models Trained

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 84.65% |
| Decision Tree | 87.82% |
| K-Nearest Neighbours | 84.57% |
| **Random Forest** | **89.06%** ✅ |

### Random Forest Configuration
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

---

## 📈 Results

- **Best Model**: Random Forest — **89.06% accuracy**, **AUC = 0.96**
- Top predictive features: `dislikes`, `likes`, `comment_count`, `country_enc`, `dislike_ratio`
- Ratio-based features outperform raw counts as they normalise for video age and audience size
- False negatives (viral videos missed) slightly exceed false positives — expected for breakout content prediction

---

## ⚠️ Limitations

- 58% of videos appear only once in the raw dataset, so for those, the earliest snapshot is also the peak snapshot — limiting true predictive horizon
- Dislikes are unavailable on YouTube post-November 2021, reducing direct applicability to current data
- Dataset covers only four countries and a limited time window

---

## 🔮 Future Work

- **NLP on titles/tags**: TF-IDF or BERT embeddings to capture semantic content signals
- **Multi-class virality tiers**: Moderately popular → Highly popular → Mega-viral
- **Dislike-free feature sets**: Adapt model for post-2021 YouTube API constraints
- **Channel-level features**: Subscriber count, upload frequency, historical performance

---

## 👥 Team

Swarnojjwal Guha · Nitesh Sharma· Tarini Parashar

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Run the notebook:
```bash
jupyter notebook youtube_virality_predictor.ipynb
```
