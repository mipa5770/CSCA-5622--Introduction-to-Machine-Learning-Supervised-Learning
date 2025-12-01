# Bacterial Species Classification Using DNA Oligomer Frequencies

Master's degree coursework project for **Introduction to Supervised Learning (CSCA-5622)** at the University of Colorado Boulder.

  
📁 **GitHub Repository:** https://github.com/mipa5770/CSCA-5622--Introduction-to-Machine-Learning-Supervised-Learning
📊 **Kaggle Dataset:**  
Kaggle. (2022). *Tabular Playground Series - Feb 2022* [Dataset]. 
Retrieved from https://www.kaggle.com/c/tabular-playground-series-feb-2022/data

**Youtube:** https://youtu.be/aQ7TaxjmSOE

---

## Project Overview

This project showcases supervised learning techniques and models to perform **multi-class classification** on bacterial DNA data. The implementation includes comprehensive data preprocessing, exploratory data analysis, model training, hyperparameter optimization, and performance evaluation.

**Competition Result:** Achieved **97.7% accuracy** on the Kaggle leaderboard, placing in the **top 30%** of competitors.

### Problem Statement

This project addresses the challenge of **automated bacterial species identification** from DNA oligomer frequency measurements. The goal is to classify bacterial samples into one of **10 bacterial species** based on 286 DNA sequence pattern features.

### Motivation and Goals

- **Clinical Relevance:** Rapid bacterial identification is critical for treatment decisions, outbreak tracking, and antibiotic resistance monitoring. Traditional culture-based methods take 24-48 hours; machine learning could reduce this to hours.
- **Learning Objectives:** Explore and compare multiple classification algorithms (Random Forest, SVM, Neural Networks) on a large-scale biological dataset.
- **Competition Goal:** Achieve high accuracy on the Kaggle Tabular Playground Series (February 2022) competition.

### Type of Learning & Task

| Attribute | Description |
|-----------|-------------|
| **Learning Type** | Supervised Learning |
| **Task Type** | Multi-class Classification (10 classes) |
| **Algorithms Used** | Random Forest, Support Vector Machine (One-vs-Rest), Deep Neural Network |

---

## Data Citation

**Original Research:**  
Wood, R. L., Jensen, T., Wadsworth, C., Clement, M., Nagpal, P., & Pitts, W. G. (2020). Analysis of Identification Method for Bacterial Species and Antibiotic Resistance Genes Using Optical Data From DNA Oligomers. *Frontiers in Microbiology*, 11, 257. https://doi.org/10.3389/fmicb.2020.00257

**Kaggle Dataset:**  
Kaggle. (2022). *Tabular Playground Series - Feb 2022* [Dataset]. Retrieved from https://www.kaggle.com/c/tabular-playground-series-feb-2022/data

---

## Dataset Description

The data originates from research on bacterial identification using optical data from DNA oligomers. The Kaggle team added simulated measurement errors to make the problem more challenging.

| Dataset | Rows | Columns | Size | Description |
|---------|------|---------|------|-------------|
| `train.csv` | 200,000 | 287 | ~1 GB | 286 float features + 1 target (string) |
| `test.csv` | 100,000 | 286 | ~500 MB | 286 float features (no target) |
| `sample_submission.csv` | 100,000 | 2 | Small | row_id + predicted target |

### Features

- **286 numeric columns** (float64) representing DNA oligomer frequencies
- Feature names follow the pattern `AxTyGzCw` representing counts of Adenine, Thymine, Guanine, and Cytosine bases in 10-mer oligomers
- Values represent frequency measurements with simulated noise

### Target Variable

The 10 bacterial species to classify:

1. *Bacteroides fragilis*
2. *Campylobacter jejuni*
3. *Enterococcus hirae*
4. *Escherichia coli*
5. *Escherichia fergusonii*
6. *Klebsiella pneumoniae*
7. *Salmonella enterica*
8. *Staphylococcus aureus*
9. *Streptococcus pneumoniae*
10. *Streptococcus pyogenes*

### Class Distribution

Nearly balanced distribution (~9.9% to 10.1% per class) — no resampling techniques required.

---

## Repository Structure

```
bacterial-classification/
├── Final-random-forest-updated.ipynb           # Main analysis notebook with outputs
├── train.csv                                   # Training data (not included - download from Kaggle)
├── test.csv                                    # Test data (not included - download from Kaggle)
├── sample_submission.csv                       # Submission format template (not included - download from Kaggle)
├── submission.csv                              # Final predictions for Kaggle
├── README.md                                   # This file
├── output.png                                  # EPOUCH output 1
├── output2.png                                 # EPOUCH output 2
```

### Key Files

- **[random-forest.ipynb](random-forest.ipynb):** Main analysis notebook with complete EDA, model training, evaluation, and results
- **[submission.csv](submission.csv):** Final Kaggle submission achieving 97.7% accuracy

---

## Methodology

### Data Cleaning

1. **Missing Values:** Verified no missing values in train or test sets
2. **Duplicates:** Identified and removed duplicate entries from training data to prevent data leakage
3. **Data Types:** All 286 features confirmed as float64, target as string (categorical)

**Summary:** Dataset was relatively clean. Duplicate removal was the primary cleaning step performed.

### Exploratory Data Analysis

- **Target Distribution:** Histogram visualization confirms balanced classes
- **Correlation Analysis:** Heatmap reveals patterns of correlated features among DNA oligomers (biologically expected as certain sequences co-occur)
- **Feature Statistics:** Examined min/max values, standard deviations across all 286 features
- **Unique Values:** Up to 12,494 unique values per feature

### Correlation and Multicollinearity

The correlation heatmap reveals visible patterns and blocks of correlated features. This is biologically expected since certain DNA sequences tend to co-occur.

**Impact on Model Selection:**
- **Random Forest:** Robust to multicollinearity because it randomly selects feature subsets for each tree
- **SVM:** Can be affected, but the RBF kernel handles this reasonably well
- **Neural Network:** Normalization layer and dropout help manage correlated inputs

Since Random Forest inherently handles correlated features through random feature selection at each split, all 286 features were retained without dimensionality reduction.

---

## Models Implemented

### 1. Random Forest Classifier

- **Approach:** Ensemble method using bootstrap aggregation (bagging)
- **Hyperparameter Tuning:** GridSearchCV with cross-validation
- **Parameters Tuned:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Final Configuration:** 300 trees with optimized depth and split parameters
- **Result:** **Best performing model** — 97.7% accuracy on Kaggle test set

### 2. Support Vector Machine (One-vs-Rest)

- **Approach:** Multi-class classification using One-vs-Rest strategy with RBF kernel
- **Hyperparameter Tuning:** GridSearchCV for `C` and `gamma` parameters
- **Challenges:** Computationally expensive on large dataset (200K samples)
- **Result:** Competitive accuracy but significantly longer training time

### 3. Deep Neural Network (TensorFlow/Keras)

- **Architecture:** 
  - Input layer (286 features)
  - Normalization layer
  - Dense hidden layers with ReLU activation
  - Dropout layers for regularization
  - Softmax output layer (10 classes)
- **Training:** 1500 epochs on full dataset
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Result:** Good performance but did not exceed Random Forest accuracy

### Training Techniques Used

| Technique | Description |
|-----------|-------------|
| Cross-Validation | 5-fold CV for model evaluation |
| GridSearchCV | Exhaustive hyperparameter search |
| Label Encoding | Convert string labels to integers (0-9) |
| Dropout | Regularization in neural network |
| Early considerations | Class balance checked (no SMOTE needed) |

---

## Results

### Model Comparison

| Model | Cross-Val Accuracy | Training Time | Notes |
|-------|-------------------|---------------|-------|
| **Random Forest** | ~97.7% | Moderate | Best overall performance |
| SVM (One-vs-Rest) | ~95% | Very Long | Computationally expensive |
| Neural Network | ~96% | Long (GPU recommended) | Required extensive tuning |

### Best Model: Random Forest

- **Kaggle Public Leaderboard:** 97.736% accuracy
- **Ranking:** Top 30% of competition participants
- **Confusion Matrix:** Strong diagonal dominance with minimal misclassifications

### Key Findings

1. **Random Forest superiority:** Tree-based ensemble methods excelled on this high-dimensional tabular data
2. **SVM limitations:** One-vs-Rest SVM was computationally prohibitive for 200K samples with 286 features
3. **Neural Network challenges:** Required careful architecture design; prone to overfitting without sufficient regularization
4. **Feature importance:** Certain oligomer frequencies showed higher predictive power (extractable from RF feature importances)

---

## Evaluation Metrics

- **Primary Metric:** Accuracy (appropriate for balanced multi-class classification)
- **Confusion Matrix:** Visualized per-class performance
- **Cross-Validation:** 5-fold CV to ensure robust generalization estimates
- **Training/Validation Loss:** Monitored for neural network to detect overfitting

---

## Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~4GB RAM minimum (dataset is large)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/mipa5770/CSCA-5622--Introduction-to-Machine-Learning-Supervised-Learning
cd bacterial-classification
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download data from Kaggle:**
   - Visit: https://www.kaggle.com/c/tabular-playground-series-feb-2022/data
   - Download `train.csv`, `test.csv`, and `sample_submission.csv`
   - Place files in the repository root directory

### Running the Notebook

```bash
jupyter notebook random-forest.ipynb
```

**Note:** Full model training takes significant time:
- Random Forest: ~10-30 minutes
- SVM: Several hours
- Neural Network: ~1-2 hours (faster with GPU)

Pre-trained models and outputs are preserved in the notebook for review without re-running.

---

## Dependencies

Core libraries used in this project:

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.4.0 | Data manipulation and analysis |
| numpy | >= 1.21.0 | Numerical computing |
| scikit-learn | >= 1.0.0 | ML algorithms, preprocessing, evaluation |
| tensorflow | >= 2.8.0 | Deep neural network implementation |
| matplotlib | >= 3.5.0 | Data visualization |
| seaborn | >= 0.11.0 | Statistical visualizations |
| joblib | >= 1.1.0 | Model persistence |

See [requirements.txt](requirements.txt) for complete dependency list.

---

## Discussion & Conclusion

### What Worked Well

- **Random Forest** proved highly effective for this high-dimensional classification task
- **Minimal preprocessing** required due to clean dataset and balanced classes
- **GridSearchCV** significantly improved model performance through systematic hyperparameter optimization

### Challenges Encountered

- **Computational constraints:** Large dataset (200K × 287) made SVM training very slow
- **Neural network tuning:** Required experimentation to find architecture that didn't overfit
- **Memory management:** Had to be mindful of RAM usage when loading full dataset

### Why Random Forest Outperformed

1. **Handles high dimensionality:** 286 features managed well through random subspace selection
2. **Robust to noise:** Ensemble averaging reduces impact of simulated measurement errors
3. **No feature scaling required:** Tree-based methods are scale-invariant
4. **Handles correlated features:** Random feature selection at each split naturally manages multicollinearity

### Future Improvements

- **Feature engineering:** Create interaction features or domain-specific transformations
- **Ensemble methods:** Combine predictions from RF, XGBoost, and NN (stacking)
- **Neural architecture search:** Experiment with deeper networks, residual connections, batch normalization
- **Dimensionality reduction:** Apply PCA or feature selection to reduce computational cost for SVM

---

## Academic Information

| Field | Details |
|-------|---------|
| **Course** | DTSA-5509: Introduction to Machine Learning — Supervised Learning |
| **Institution** | University of Colorado Boulder |
| **Program** | Master of Science in Data Science |
| **Term** | [Your Term Here] |

---

## Acknowledgments

- **University of Colorado Boulder** — Course instruction and academic support
- **Kaggle** — For hosting the Tabular Playground Series competition
- **Original Researchers** — Wood et al. for the foundational bacterial identification research
- **Course Instructors** — For guidance on supervised learning techniques

---

## License

This project is intended for **academic purposes only**. The dataset is publicly available through Kaggle's Tabular Playground Series. Please cite the original research paper if using insights from this analysis.

---

## Contact

For questions or collaboration opportunities:
- **GitHub:** https://github.com/mipa5770/CSCA-5622--Introduction-to-Machine-Learning-Supervised-Learning
- **Email:** mipa5770@colorado.edu 
