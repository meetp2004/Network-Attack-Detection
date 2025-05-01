# Network Attack Detection using Machine Learning

## Model Architecture

### 1. Hierarchical SVM
- Level 1: Binary classification (Benign vs Attack)
- Level 2: Attack group classification (DoS/DDoS, Brute Force, Reconnaissance)
- Level 3: Specific attack classification within each group
  - DoS/DDoS: GoldenEye, Hulk, Slowhttptest, Slowloris, DDoS
  - Brute Force: FTP-Patator, SSH-Patator, Heartbleed, Bot
  - Reconnaissance: PortScan, Infiltration

### 2. Random Forest
- Multiple configurations tested with different parameters:
  - Number of estimators: 100, 150, 200
  - Criterion: Entropy and Gini
- Feature scaling using StandardScaler
- SMOTE for handling class imbalance

### 3. CNN
- 5 Convolutional layers with batch normalization
- Dropout layers for regularization
- Dense layers for final classification
- Early stopping and learning rate scheduling
  
## Installation

1. Clone the repository
2. Install the required dependencies
### Training
Train all models by running training.py

### Testing
Test all models by running testing.py

The models are evaluated using metrics including:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Performance metrics are generated during this step.

## Dataset Used
- CICIDS 2017
