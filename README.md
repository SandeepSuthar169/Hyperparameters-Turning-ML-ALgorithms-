
# Hyperparameters Tuning for Logistic Regression, SVC, KNN, RandomForest, Decision Tree 

This notebook demonstrates the process of hyperparameter tuning for three machine learning models: Logistic Regression, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), RandomForest, Decision Tree using the GridSearchCV method. The dataset used is the heart disease dataset (heart.csv), which contains medical records of patients and aims to predict the presence of heart disease.

## Dataset Overview
- Shape: 303 rows, 14 columns

 - Features: Age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise-induced angina (exang), ST depression induced by exercise relative to rest (oldpeak), slope of the peak exercise ST segment (slope), number of major vessels (ca), thalassemia type (thal), and target (presence of heart disease).

- Target Variable: target (0 = no heart disease, 1 = heart disease)
## Project Structure
``` bash 

├── data/		       # Raw and processed datasets (heart.csv)
├── Hyperparameters_Tunning (DT, RF)/    # Vscode and  Colab for analysis and model building
├── Hyperparameters_Tunning (Lg, SVM, NKK)/ # Vscode and  Colab for analysis and model building
├── .gitignore/	     # installing libraries file
├── README.md	    # Project description

```
## Models Tuned
1. Logistic Regression 
- Parameters tuned : `penalty`, `dual`, `C`, `class_weight`, `solver`, `random_state`, `max_iter`

- Best Parameters found :`{'C': 1.0, 'class_weight': 'balanced', 'dual': False, 'max_iter': 10, 'penalty': 'l2', 'random_state': None, 'solver': 'liblinear'}`

2. Support Vector Classifier(SVC)
- Parameters tuned : `C`, `kernel`, `degree`, `gamma`, `coef0`, `class_weight`
 
 - Best Parameters found :    `{'C': 1.0, 'class_weight': None, 'coef0': 0.000', degree'1', 'gamma': 'scale', 'kernel': 'linear'}`

3. K-NearestNeighbors (NKK)
- Parameters tuned :`n_neighbors`, `metrics`, `p`, `weights`

- Best Parameters found : `{'n_neighbors' : 7, 'metrics' : 'minkowski', 'p' : 1, 'weights' : 'distance'}`

4. Decision Tree (DT)
- Parameters tunde : `criterion`, `splitter`, `max_depth`, `min_sample_split`, `min_sample_leaf`, `max_features`, `max_leaf_nodes`

- Best Parameters found :`{'criterion': 'entropy','max_depth': 10'max_features':'sqrt',max_leaf_nodes': None,'min_samples_leaf': 1'min_samples_split': 2,'splitter': 'random'}`

5. Random Forest
- parameters tuned : `n_neighbors`, `max_features`, `max_depth`, `min_sample_split`, `min_sample_lead`, `class_weight`, `max_leaf_modes`, `oob_score`, `bootstrap`, `criterion`

- Best Parameters found : `{'bootstrap':False,'class_weight':'balanced','criterion': 'gini','max_depth': 30', max_features':'log2',max_leaf_nodes': None',min_samples_leaf':3,'min_samples_split':4,'n_estimators': 50,'oob_score': False}`

6. SGDClassifier
   - parameters tuned : `loss`, `penalaty`, `alpha`, `l1_ratio`, `fit_intercept`, `max_iter`, `tol`, `shuffle`, `ela0`, `epsilon`, `n_iter_no_change`, `early_stopping`
  
   - Best Parameters found : `{'early_stopping' : False, 'fit_intercept' : False,'l1_ratio' : 0.15,'loss' : 'log_loss' , 'max_iter': 1000, 'n_iter_no_change': 10, 'n_jobs': -1, 'penalty': 'l2','shuffle': True,'tol': 0.001}`

## Dependencies
- Python 3.10
- Libraries : `numpy`, `pandas`, `matplotlib`, `seaborn`, `scilit-learn`


## key Takeaways
- Logistic Regression performed well ( ~83.5% CV accuracy) with the given parameters.
- SVM  performed well ( ~81.4%  accuracy ) with the given parameters.
- K-NN performed well ( ~70.23%  accuracy ) with the given parameters.
- Decisioin Tree performed well ( ~81.78%  accuracy ) with the given parameters.
- Random Forest performed well (~84.73%  accuracy ) with the given parameters.
- SGDClassifier perform well (~83.63  accuracy ) with the given parameters.
