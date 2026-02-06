# 🛡️ Network Intrusion Detection Using Machine Learning

<div align="center">

**A comprehensive machine learning solution for detecting anomalous network traffic**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green.svg)](https://scikit-learn.org/)

</div>

---

## 📖 About This Project

This project implements a **production-ready machine learning pipeline** for network intrusion detection. It classifies network traffic as either **normal** or **anomaly** using classical ML algorithms enhanced with modern hyperparameter optimization techniques.

### 🎯 Objective
Build an accurate binary classifier to identify malicious network connections, helping cybersecurity systems detect and prevent intrusion attempts in real-time.

### 🌟 What Makes This Project Special?

- ✅ **End-to-End ML Pipeline**: From raw data to production-ready models
- ✅ **Smart Feature Engineering**: RFE-based feature selection reduces 40+ features to top 10
- ✅ **Hyperparameter Optimization**: Optuna framework for automated model tuning
- ✅ **Robust Evaluation**: 10-fold cross-validation ensures generalization
- ✅ **Multiple Algorithms**: Comparative analysis of 3 different ML approaches
- ✅ **Clean & Documented**: Professional notebook structure with clear explanations

---

## 📊 Dataset Information

| Property | Details |
|----------|---------|
| **Source** | Network Intrusion Detection Challenge |
| **Files** | `Train_data.csv`, `Test_data.csv` |
| **Target Variable** | `class` (binary: normal / anomaly) |
| **Features** | 40+ network connection attributes |
| **Feature Types** | Protocol type, service, bytes transferred, duration, flags, etc. |
| **Task Type** | Binary Classification (Supervised Learning) |

---

## 🔬 Methodology & Workflow

### **Phase 1: Data Exploration** 🔍
- Load training and test datasets from Google Drive
- Analyze data structure, types, and statistical summaries
- Identify missing values and check for duplicates
- Visualize class distribution to assess balance

### **Phase 2: Data Preprocessing** 🧹
1. **Label Encoding**: Convert categorical features to numerical (train+test together for consistency)
2. **Data Cleaning**: Remove zero-variance columns (`num_outbound_cmds`)
3. **Feature Selection**: Apply Recursive Feature Elimination (RFE) with Random Forest
   - Reduce from 40+ features to **top 10 most important features**
4. **Standardization**: Apply StandardScaler for distance-based algorithms
5. **Train-Validation Split**: 70% training, 30% validation

### **Phase 3: Model Training** 🤖
Three algorithms evaluated:

| Model | Description | Optimization |
|-------|-------------|--------------|
| **Logistic Regression** | Linear baseline model | Default parameters |
| **K-Nearest Neighbors (KNN)** | Instance-based learning | Optuna tuning (n_neighbors) |
| **Decision Tree** | Tree-based classifier | Optuna tuning (max_depth, max_features) |

### **Phase 4: Evaluation & Validation** 📈
- **Training/Test Accuracy**: Measure overfitting and generalization
- **10-Fold Cross-Validation**: Compute precision & recall across folds
- **Confusion Matrices**: Analyze false positives and false negatives
- **Classification Reports**: Detailed precision, recall, F1-score per class
- **F1-Score Comparison**: Visual bar charts for model comparison

---

## 🏆 Results & Performance

### Model Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score | Optimization |
|-------|--------------|-----------|--------|----------|--------------|
| **Logistic Regression** | 92% | 0.92 | 0.92 | 0.92 | None (default) |
| **KNN (Tuned)** | 98% | 0.98 | 0.98 | 0.98 | Optuna optimized `n_neighbors` |
| **Decision Tree (Tuned)** | **99.5%** | 0.99 | 1.00 | 1.00 | Optuna optimized `max_depth`, `max_features` |

### Detailed Model Performance

#### **🥇 Decision Tree (Best Performer)**
- **Test Accuracy**: 99.5%
- **Precision**: 0.99 (normal) | 1.00 (anomaly)
- **Recall**: 1.00 (normal) | 0.99 (anomaly)
- **F1-Score**: 0.99 (normal) | 1.00 (anomaly)
- **Confusion Matrix**: [[3484, 14], [23, 4037]]
  - Only 14 false positives and 23 false negatives out of 7,558 test samples

#### **🥈 K-Nearest Neighbors (Tuned)**
- **Test Accuracy**: 98.3%
- **Precision**: 0.98 (both classes)
- **Recall**: 0.98 (both classes)
- **F1-Score**: 0.98 (both classes)
- **Confusion Matrix**: [[3435, 63], [65, 3995]]
  - 128 total misclassifications out of 7,558 samples

#### **🥉 Logistic Regression (Baseline)**
- **Test Accuracy**: 92.3%
- **Precision**: 0.94 (normal) | 0.91 (anomaly)
- **Recall**: 0.89 (normal) | 0.95 (anomaly)
- **F1-Score**: 0.92 (normal) | 0.93 (anomaly)
- **Confusion Matrix**: [[3129, 369], [211, 3849]]
  - Good baseline but higher error rate compared to tuned models

### Key Performance Metrics

#### ✅ **Evaluation Metrics Explained**
- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted anomalies are actually anomalies (minimize false alarms)
- **Recall**: How many actual anomalies are detected (critical for security - minimize missed intrusions)
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)

#### 📊 **Cross-Validation Results**
- **10-Fold Cross-Validation** performed for all models
- Consistent scores across folds indicate good generalization
- No significant overfitting observed

### 🎯 Key Findings

1. **Feature Selection Impact**: RFE reduced dimensionality significantly while maintaining model performance
2. **Hyperparameter Tuning**: Optuna optimization improved KNN and Decision Tree accuracy by tuning critical parameters
3. **Model Generalization**: All models showed consistent performance on cross-validation
4. **Best Performers**: Tuned KNN and Decision Tree models achieved superior results compared to baseline Logistic Regression

---

## 🛠️ Technologies & Tools

**Programming Language**
- Python 3.7+

**Core Libraries**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and utilities

**Visualization**
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization

**Machine Learning**
- `LogisticRegression` - Linear classification
- `KNeighborsClassifier` - Instance-based learning
- `DecisionTreeClassifier` - Tree-based learning
- `RandomForestClassifier` - Ensemble method (used in RFE)
- `StandardScaler` - Feature standardization
- `LabelEncoder` - Categorical encoding

**Optimization**
- `optuna` - Automated hyperparameter tuning framework

**Additional**
- `xgboost`, `lightgbm` - Advanced gradient boosting (imported for extensibility)
- `tabulate` - Pretty table formatting

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab account
- Basic understanding of machine learning concepts

### Option 1: Google Colab (Recommended) ☁️

**Perfect for quick start without local setup**

1. **Upload Notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `network_intrusion_detection.ipynb`

2. **Upload Datasets**
   - Upload `Train_data.csv` and `Test_data.csv` to your Google Drive
   - Recommended location: `MyDrive/` (root of Google Drive)

3. **Configure File Paths**
   - Update these lines in the notebook (Cell 4):
   ```python
   train = pd.read_csv('/content/drive/MyDrive/Train_data.csv')
   test = pd.read_csv('/content/drive/MyDrive/Test_data.csv')
   ```

4. **Run All Cells**
   - Click `Runtime > Run all` to execute the entire pipeline

### Option 2: Local Jupyter Notebook 💻

**For users who prefer local development**

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/network-intrusion-detection.git
   cd network-intrusion-detection
   ```

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r network_intrusion_requirements.txt
   ```

4. **Place Datasets**
   - Put `Train_data.csv` and `Test_data.csv` in the project directory
   - Update file paths in the notebook accordingly

5. **Launch Notebook**
   ```bash
   jupyter notebook network_intrusion_detection.ipynb
   ```

---

## 📁 Project Structure

```
network-intrusion-detection/
│
├── 📓 network_intrusion_detection.ipynb   # Main Jupyter notebook
│
├── 📊 Train_data.csv                      # Training dataset
├── 📊 Test_data.csv                       # Testing dataset
│
├── 📄 intrusion_detection_README.md       # Project documentation (this file)
├── 📄 network_intrusion_requirements.txt  # Python dependencies
├── 📄 intrusion_detection_gitignore.txt   # Git ignore rules (rename to .gitignore)
│
└── 📄 GITHUB_UPLOAD_CHECKLIST.txt         # GitHub upload guide (reference only)
```

---

## 📝 Notebook Structure

The notebook is organized into **8 main sections**:

### 1️⃣ Import Libraries
- Load all required Python packages
- Configure warnings and display settings

### 2️⃣ Load Data
- Mount Google Drive (for Colab)
- Load training and test CSV files
- Display dataset shapes

### 3️⃣ Exploratory Data Analysis (EDA)
- Examine data structure and types
- Statistical summaries
- Missing value analysis
- Class distribution visualization

### 4️⃣ Data Preprocessing (5 subsections)
- **4.1**: Label encoding for categorical features
- **4.2**: Data cleaning (remove zero-variance columns)
- **4.3**: Feature selection using RFE (Recursive Feature Elimination)
- **4.4**: Feature standardization with StandardScaler
- **4.5**: Train-validation split (70-30)

### 5️⃣ Model Training (4 subsections)
- **5.1**: Logistic Regression (baseline)
- **5.2**: K-Nearest Neighbors with Optuna tuning
- **5.3**: Decision Tree with Optuna tuning
- **5.4**: Model comparison table

### 6️⃣ Cross-Validation (4 subsections)
- **6.1**: 10-fold cross-validation setup
- **6.2**: Compute precision scores
- **6.3**: Compute recall scores
- **6.4**: Results summary

### 7️⃣ Final Testing (4 subsections)
- **7.1**: Train models on full training set
- **7.2**: Generate predictions on test set
- **7.3**: Confusion matrices and classification reports
- **7.4**: F1-score comparison visualization

### 8️⃣ Conclusion
- Summary of findings
- Model performance overview
- Key takeaways

---

## ⚙️ Configuration & Customization

### Adjusting Optuna Trials

**KNN Optimization** (Currently: 1 trial for demo)
```python
study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=1)  # Increase to 50-100 for better results
```

**Decision Tree Optimization** (Currently: 30 trials)
```python
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=30)  # Increase to 100+ for thorough search
```

### Feature Selection

**Changing Number of Features**
```python
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)  # Change 10 to desired number
```

### Train-Test Split Ratio
```python
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3)  # Change 0.3 to desired ratio
```

---

## 🔍 Understanding the Output

### Confusion Matrix Interpretation
```
[[True Negatives   False Positives]
 [False Negatives  True Positives]]
```

- **True Negatives (TN)**: Normal traffic correctly classified as normal
- **False Positives (FP)**: Normal traffic incorrectly flagged as anomaly
- **False Negatives (FN)**: Anomaly missed (classified as normal) ⚠️ **Critical in security**
- **True Positives (TP)**: Anomaly correctly detected ✅

### Classification Report Metrics

```
              precision    recall  f1-score   support
     normal       0.95      0.98      0.97      5000
    anomaly       0.92      0.85      0.88      2000
```

- **Precision**: Of all predicted anomalies, how many are actually anomalies?
- **Recall**: Of all actual anomalies, how many did we detect?
- **F1-Score**: Balanced metric (harmonic mean of precision and recall)
- **Support**: Number of samples in each class

---

## 💡 Key Insights & Learnings

### 1. Feature Engineering Matters
- **Before RFE**: 40+ features (high dimensionality, potential noise)
- **After RFE**: 10 features (reduced complexity, maintained performance)
- **Impact**: Faster training, better generalization, easier interpretation

### 2. Hyperparameter Tuning Delivers Results
- **Optuna Framework**: Automated search for optimal parameters
- **KNN**: Found optimal `n_neighbors` value
- **Decision Tree**: Optimized `max_depth` and `max_features`
- **Result**: Significant performance improvement over default parameters

### 3. Cross-Validation Confirms Robustness
- **10-Fold CV**: Model tested on multiple data splits
- **Consistent Scores**: Indicates stable performance
- **No Overfitting**: Models generalize well to unseen data

### 4. Model Selection Trade-offs
- **Logistic Regression**: Fast, interpretable, good baseline
- **KNN**: High accuracy, but slower on large datasets
- **Decision Tree**: Interpretable, handles non-linear patterns well

### 5. Security-Focused Metrics
- **Recall is Critical**: Missing an intrusion (false negative) is costly
- **Precision Matters**: Too many false alarms reduce trust
- **F1-Score Balance**: Optimal trade-off between precision and recall

---

## 🚧 Future Enhancements

- **Ensemble Methods**: Implement Random Forest and XGBoost for improved accuracy
- **Neural Networks**: Experiment with deep learning approaches (LSTM, CNN)
- **Model Explainability**: Add SHAP values for feature importance analysis
- **Real-time Deployment**: Create REST API endpoint for production use
- **Multiclass Classification**: Extend to detect specific attack types (DoS, Probe, R2L, U2R)
- **Imbalanced Data Handling**: Implement SMOTE or class weighting if needed
- **Time Series Analysis**: Incorporate temporal patterns in network traffic sequences
