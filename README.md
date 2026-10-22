# 🧪 Data Science & AI Lab

> Learn ML, Deep Learning, GANs, NLP and Computer Vision by **testing hypotheses, not tuning accuracy**.
> Every project starts with a plain-English analogy, states a hypothesis, backs it with real statistics, and ends with a chart that tells the story.

**▶ Interactive stories:** [`docs/index.html`](docs/index.html) (served by GitHub Pages: a searchable index of every report) · **Status:** 🚧 revamp in progress — projects marked ⏳ are legacy code awaiting their rewrite.

## ⚡ Quick start

```bash
git clone git@github.com:debanjansaha-git/ML_DL_Modelling.git && cd ML_DL_Modelling
uv sync --extra ml            # pick a track: ml | dl | gan | nlp | cv   (Python 3.11+)
uv run jupyter lab            # notebooks
uv run python ML/<project>/main.py
```

Data and model weights are **not** stored in git. Each project downloads what it needs into its own `data/` folder (gitignored) on first run.

## 🗺️ Curriculum map

**Suggested path:** 🟢 ML foundations → 🟡 applied ML → 🟡 deep learning → 🔴 NLP / GANs / CV

### 1 · Machine Learning (`ML/`)
| Level | Project | You'll learn | Status |
|---|---|---|---|
| 🟢 | [Haberman Survival](ML/haberman-survival) | EDA, Mann-Whitney U, chi-square, leakage-safe CV | ✅ |
| 🟢 | [Diabetes Risk](ML/diabetes-risk) | Spearman, Kruskal-Wallis, odds ratios, collinearity, honest CV | ✅ |
| 🟢 | [Birth Rate Analysis](ML/birth-rate-analysis) | Calendar effects, paired and rank tests, rolling-origin forecasting | ✅ |
| 🟢 | [Breast Cancer Wisconsin](ML/breast-cancer-wisconsin) | Effect sizes, collinearity, nested CV, sensitivity-vs-false-alarm trade-off | ✅ |
| 🟡 | [Graduate Admission](ML/graduate-admission) | Confounding, collinearity, robust regression, permutation importance | ✅ |
| 🟡 | [Housing Price Prediction](ML/housing-price-prediction) | Log targets, partial F-test, regularisation, leak-free CV (Ames) | ✅ |
| 🟡 | [Credit Risk Analysis](ML/credit-risk-analysis) | Odds ratios, calibration, cost-based decisions, fairness audit | ✅ |
| 🟡 | [Credit Card Fraud Detection](ML/credit-card-fraud-detection) | Rare events, average precision, cost-based thresholds, time-ordered validation | ✅ |
| 🟡 | [IPL Score Prediction](ML/ipl-score-prediction) | Time-aware evaluation, drift, cluster bootstrap, prediction intervals | ✅ |
| 🔴 | [Delhi Weather ARIMA](ML/delhi-weather-arima) | Trend tests, stationarity, AR forecasting skill by horizon | ✅ |

### 2 · Deep Learning (`DL/`, PyTorch)
| Level | Project | You'll learn | Status |
|---|---|---|---|
| 🟢 | [Linear Regression in PyTorch](DL/linear-regression-pytorch) | Gradient descent, learning-rate limits, robust loss | ✅ |
| 🟢 | [Adam Optimizer](DL/adam-optimizer) | Adam from scratch, fair optimiser races, learning-rate sensitivity | ✅ |
| 🟡 | [Housing Regression NN](DL/housing-regression-nn) | Standardising, tree baselines, architecture, spatial errors | ✅ |
| 🟡 | [MNIST Multiclass](DL/mnist-multiclass) | Softmax from scratch, calibration, shift robustness | ✅ |
| 🟡 | [CIFAR-10 CNN](DL/cifar10-cnn) | Model ladder, augmentation, McNemar tests, confusion analysis | ✅ |
| 🟡 | [Malaria Cell Classification](DL/malaria-cell-classification) | Colour baseline, transfer learning, slide-level splits | ✅ |
| 🟡 | [Skin Cancer (HAM10000)](DL/skin-cancer-lesion-classification) | Leakage from resampling, imbalance, melanoma trade-offs | ✅ |
| 🔴 | [Bidirectional LSTM](DL/bidirectional-lstm) | Readout bug, direction vs capacity, row order | ✅ |
| 🔴 | [Stock Price LSTM](DL/stock-price-lstm) | Random walk, persistence baseline, Diebold-Mariano | ✅ |
| 🔴 | [ECG Anomaly Detection](DL/ecg-anomaly-detection) | Autoencoders, baselines, thresholds, dirty training data | ✅ |
| 🔴 | [RL Shower Temperature](DL/rl-shower-temperature) | Exact solution, rescaling, table vs DQN, extrapolation | ✅ |

### 3 · Generative Models (`GAN/`)
| Level | Project | You'll learn | Status |
|---|---|---|---|
| 🔴 | [DCGAN](GAN/dcgan) | Judging a GAN with a referee, recipe sensitivity, memorisation | ✅ |
| 🔴 | [WGAN](GAN/wgan) | A loss that tracks quality, and how touchy weight clipping is | ✅ |
| 🔴 | [WGAN-GP](GAN/wgan-gp) | Gradient penalty vs clipping, penalty weight and why not batch norm | ✅ |

### 4 · Natural Language Processing (`NLP/`, 🤗 Transformers)
| Level | Project | You'll learn | Status |
|---|---|---|---|
| 🟢 | [Spam Detection (TF-IDF)](NLP/spam-detection-tfidf) | TF-IDF, leakage checks, disguised spam | ✅ |
| 🟡 | [SMS Spam with DistilBERT](NLP/sms-spam-distilbert) | Fine-tuning vs baseline | ⏳ |
| 🟡 | [Hotel Review Sentiment](NLP/hotel-review-sentiment) | Text classification at scale | ⏳ |
| 🟡 | [Stock News Direction](NLP/stock-news-direction) | A split bug behind an 86% accuracy, base rates | ✅ |
| 🟡 | [Multilingual Sentiment (BERT)](NLP/multilingual-sentiment-bert) | Multilingual transfer | ⏳ |
| 🟡 | [Text Summarization](NLP/text-summarization) | Extractive vs abstractive | ⏳ |
| 🔴 | [Text Generation](NLP/text-generation) | Markov chains: fluency vs copying, vs a neural model | ✅ |
| 🔴 | [Question Answering](NLP/question-answering) | Extractive QA | ⏳ |

### 5 · Computer Vision (`CV/`, OpenCV + MediaPipe)
| Level | Project | You'll learn | Status |
|---|---|---|---|
| 🟢 | [Document Scanner](CV/document-scanner) | Measuring a classic OpenCV recipe on synthetic photos with known truth | ✅ |
| 🟢 | [Haar Face Detection](CV/haar-face-eye-detection) | The classic cascade vs modern detectors on WIDER FACE | ✅ |
| 🟢 | [Haar Face & Eye Detection](CV/haar-face-eye-detection) | Classic detectors | ⏳ |
| 🟢 | [Virtual Painter](CV/virtual-painter) | Tracking a coloured pen under changing light, with exact ground truth | ✅ |
| 🟡 | [Hand Detection](CV/hand-detection) | How size, zoom and confidence decide what MediaPipe finds | ✅ |
| 🟡 | [Gesture Volume Controller](CV/gesture-volume-controller) | Landmarks → control signal | ⏳ |
| 🔴 | [Object Detection](CV/object-detection) | Modern detectors | ⏳ |
| 🔴 | [Face Recognition](CV/face-recognition) | Embeddings and thresholds | ⏳ |
| 🔴 | [Virtual Attendance](CV/virtual-attendance) | End-to-end face pipeline | ⏳ |

## 🧭 How every project is laid out

```
<project>/
├── README.md     # 💡 analogy · 🔬 hypothesis · 📊 insights · ⚙️ under the hood · 🚀 run · ⚠️ limitations
├── main.py       # or a notebook: problem → hypothesis → data → test → baseline → iteration → takeaways
├── data/         # gitignored; downloaded by script
└── outputs/      # saved plots and metrics (what docs/ pages show)
```

## 🛠️ Tooling
`uv` for environments · `ruff` for lint and Google-style docstrings · `pip-audit` for CVEs.

## 🤝 Contributing
Open an issue or PR. Follow the project layout above and state your null and alternative hypotheses before modelling.

## 📜 License
MIT. If this helps you learn, a ⭐ is appreciated.
