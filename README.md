# DevelopersHub-Corporation-AI_ML-Engineering-Internship-Phase-2

# AI/ML Engineering — Advanced Internship Tasks
---

## Task 1: News Topic Classifier Using BERT

### Objective
Fine-tune `bert-base-uncased` to classify news headlines into 4 topic categories using the AG News dataset.

### Methodology / Approach
1. Load AG News dataset via Hugging Face `datasets` library
2. Tokenize with `AutoTokenizer` (max_length=128, dynamic padding)
3. Fine-tune `bert-base-uncased` with `Trainer` API for 3 epochs
4. Evaluate with accuracy + weighted F1-score on held-out test set
5. Deploy interactively via **Gradio** with example headlines

### Architecture
```
Headline Text → BERT Tokenizer → bert-base-uncased
                                  ↓
                            [CLS] pooling
                                  ↓
                        Linear(768 → 4) + Softmax
                                  ↓
                   World | Sports | Business | Sci/Tech
```

### Key Results
| Metric | Value |
|--------|-------|
| Base Model | bert-base-uncased (110M params) |
| Training samples | 3,400 |
| Validation samples | 600 |
| Test samples | 800 |
| Epochs | 3 |
| Test Accuracy | ~93–95% (varies by run) |
| Test F1 (weighted) | ~93–95% |

### Key Observations
- BERT's contextual embeddings significantly outperform TF-IDF + Logistic Regression baselines
- Sports headlines are easiest to classify; World ↔ Business occasionally overlap
- 3 epochs is sufficient for convergence — additional epochs risk overfitting on small subsets
- Gradio deployment requires zero frontend code

---

## Task 2: End-to-End ML Pipeline for Customer Churn Prediction

### Objective
Build a reusable, production-ready scikit-learn `Pipeline` for predicting Telco customer churn, with `GridSearchCV` tuning and `joblib` export.

### Methodology / Approach
1. Load IBM Telco Churn dataset (~7,043 rows, 19 features)
2. Define separate preprocessing pipelines for numeric and categorical features via `ColumnTransformer`
3. Build end-to-end `Pipeline` objects for Logistic Regression and Random Forest
4. Tune with `GridSearchCV` (5-fold StratifiedKFold, scored on ROC-AUC)
5. Export best pipeline with `joblib` for production inference

### Pipeline Architecture
```
Raw DataFrame (no preprocessing needed at inference)
       │
       ▼
ColumnTransformer
  ├─ Numeric: SimpleImputer(median) → StandardScaler
  └─ Categorical: SimpleImputer(mode) → OneHotEncoder
       │
       ▼
Classifier (LogisticRegression | RandomForest)
       │
       ▼
Churn Probability + Binary Prediction
```

### Key Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression (tuned) | ~80% | ~0.85 |
| Random Forest (tuned) | ~82% | ~0.87 |

### Key Observations
- Month-to-month contract customers churn at 3× the rate of 2-year contract customers
- Tenure is the strongest negative predictor — long-term customers rarely churn
- Fiber optic internet users have disproportionately high churn (likely price sensitivity)
- `joblib` pipeline includes all preprocessing — a single `predict()` call handles raw data

---

## Task 3: Multimodal ML — Housing Price Prediction (Images + Tabular)

### Objective
Predict housing prices by fusing CNN-extracted visual features from house images with structured tabular property data.

### Methodology / Approach
1. Load California Housing dataset as the tabular component
2. Generate synthetic house images correlated with price (or swap in real photos)
3. Extract 512-dim features using frozen **ResNet18** (ImageNet pre-trained)
4. Reduce image features to 32 dims via **PCA**
5. Concatenate image (32-dim) + tabular (8-dim) → 40-dim fused vector
6. Train **Gradient Boosting Regressor** on tabular-only, image-only, and fused features
7. Compare MAE, RMSE, R² across all three settings

### Architecture
```
House Image (128×128)
      │
      ▼
ResNet18 (frozen, ImageNet weights)
      │
    512-dim
      │
    PCA → 32-dim ─────────────────────┐
                                       │ concat → 40-dim
Tabular (8 features) → StandardScaler ┘
                                       │
                            Gradient Boosting Regressor
                                       │
                                  Price Prediction
```

### Key Results
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Tabular only | ~$48K | ~$70K | ~0.65 |
| Image only | ~$85K | ~$115K | ~0.25 |
| **Fused (Multimodal)** | **~$44K** | **~$65K** | **~0.70** |

*(Results vary slightly per run; fused model consistently outperforms unimodal baselines)*

### Key Observations
- Tabular features (median income, lat/lon) carry the dominant price signal
- CNN image features alone achieve moderate performance, confirming visual information is informative
- The **fused multimodal model achieves the best MAE and R²** — validating the multimodal approach
- PCA on CNN features prevents the 512-dim image space from drowning out 8-dim tabular features
- With real house photos (Zillow/Redfin), image features would carry substantially more signal

---


### Running Order
| Notebook | Estimated Runtime | GPU Recommended? |
|----------|-------------------|------------------|
| Task1_News_Topic_Classifier_BERT.ipynb | 10–30 min | ✅ Yes (2-5 min on GPU) |
| Task2_ML_Pipeline_Customer_Churn.ipynb | 5–15 min | ❌ CPU sufficient |
| Task3_Multimodal_Housing_Price_Prediction.ipynb | 10–20 min | ✅ Yes (speeds up CNN) |

> **Google Colab tip:** Enable GPU via Runtime → Change runtime type → T4 GPU for Tasks 1 and 3.

---

## Repository Structure

```
.
├── Task1_News_Topic_Classifier_BERT.ipynb
├── Task2_ML_Pipeline_Customer_Churn.ipynb
├── Task3_Multimodal_Housing_Price_Prediction.ipynb
├── README.md
└── outputs/                   (auto-generated when notebooks run)
    ├── bert_news_final/       (fine-tuned BERT weights)
    ├── churn_pipeline_best.joblib
    └── *.png                  (visualisation exports)
```

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Hugging Face Transformers | BERT fine-tuning (Task 1) |
| Hugging Face Datasets | AG News loading (Task 1) |
| Gradio | Model deployment UI (Task 1) |
| scikit-learn Pipeline API | End-to-end ML pipeline (Task 2) |
| GridSearchCV | Hyperparameter tuning (Task 2) |
| joblib | Pipeline serialisation (Task 2) |
| PyTorch + torchvision | ResNet18 CNN feature extractor (Task 3) |
| PCA | Image feature dimensionality reduction (Task 3) |

---

*DevelopersHub Corporation — AI/ML Engineering Advanced Internship*
