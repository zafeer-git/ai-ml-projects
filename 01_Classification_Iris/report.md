# 🌸 Iris Flower Classification — Random Forest & Decision Tree

### 📘 Objective
Classify iris flowers into species (Setosa, Versicolor, Virginica) using supervised ML models.

### 🧠 Techniques Used
- Random Forest & Decision Tree Classifiers  
- Feature Scaling using `StandardScaler`  
- Hyperparameter tuning via `GridSearchCV`  
- Feature importance ranking  
- Visualization with confusion matrices and feature histograms  

### 📊 Key Results
| Model | Accuracy | Best Params | Key Features |
|--------|-----------|--------------|---------------|
| Random Forest | 1.00 | `{'n_estimators': 150, 'min_samples_leaf': 2}` | petal length, petal width |
| Decision Tree | 1.00 | `{'max_depth': None, 'min_samples_leaf': 4}` | petal length, sepal length |

### 🧩 Insights
Both models achieved perfect accuracy, confirming the dataset’s clear separability.  
Petal dimensions were the strongest predictors of species.

### 📈 Visuals
- `outputs/confusion_matrix_rf.png`
- `outputs/confusion_matrix_dt.png`
- `outputs/feature_importance.png`

### 🧭 Next Steps
- Add logistic regression baseline  
- Compare cross-validation folds for variance analysis  
- Deploy via a Streamlit mini demo  
