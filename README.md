# Data-Mining---First-Project
Machine Learning Algorithm Comparison for Spam Detection

### Machine Learning Algorithm Comparison for Spam Detection

This repository contains a comprehensive comparison of various machine learning algorithms for spam detection. The dataset used is the **Spambase Dataset** from the UCI Machine Learning Repository, which can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/spambase).

#### Objectives:
1. **Compare the performance** of multiple machine learning algorithms on the spam detection task.
2. **Evaluate training time** (in seconds) and **testing time** (in seconds) for each algorithm.
3. **Interpret the results** and analyze the impact of removing non-informative features on model performance.
4. **Identify the best-performing algorithms** and their optimal hyperparameters.

#### Algorithms Evaluated:
- **Naive Bayes**
- **Decision Trees**
- **Random Forest**
- **Adaboost**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

#### Key Steps:
1. **Data Preparation**:
   - Split the dataset into **training** and **testing** sets (e.g., 70% training, 30% testing).
   - Perform feature analysis to understand the relationship between features and the target variable.
   - Remove non-informative features if necessary.

2. **Model Training and Evaluation**:
   - Train each algorithm on the training set.
   - Evaluate performance on the test set using metrics such as accuracy, precision, recall, and F1-score.
   - Compare training and testing times for each algorithm.

3. **Hyperparameter Tuning**:
   - Identify the best hyperparameters for each algorithm using techniques like cross-validation.
   - Analyze the impact of hyperparameters on model performance.

4. **Results Interpretation**:
   - Compare the performance of all algorithms.
   - Highlight the top 2-3 best-performing algorithms.
   - Discuss the trade-offs between accuracy, training time, and testing time.

#### Dataset Details:
- The dataset contains **57 features** and a binary target variable indicating whether an email is spam or not.
- Features include word frequencies, character frequencies, and other email attributes.

#### Repository Structure:
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for data analysis, model training, and evaluation.
- `results/`: Performance metrics and comparison tables.
- `scripts/`: Python scripts for model training and evaluation.

#### How to Use:
1. Clone the repository.
2. Download the dataset from the provided link and place it in the `data/` folder.
3. Run the notebooks in the `notebooks/` folder to reproduce the analysis and results.

#### Dependencies:
- Python 3.x
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

#### Contribution:
Feel free to contribute by improving the code, adding new algorithms, or suggesting better hyperparameter tuning strategies. Open an issue or submit a pull request!

---

This repository is designed to help you understand the performance of different machine learning algorithms for spam detection and make informed decisions about which algorithm to use based on your specific requirements.
