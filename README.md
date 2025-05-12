
This project demonstrates the use of a machine learning pipeline to classify flower species from the Iris dataset using the **K-Nearest Neighbors (KNN)** algorithm. 
It includes data preprocessing, model tuning with grid search cross-validation, and evaluation through classification metrics and visualization.

## Overview

- **Dataset**: Iris Dataset (150 samples, 3 classes: Setosa, Versicolor, Virginica)
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Pipeline Steps**:
  - Standardize features using `StandardScaler`
  - Expand features with `PolynomialFeatures`
  - Classify with `KNeighborsClassifier`
- **Hyperparameter Tuning**: Grid search on `n_neighbors` from 1 to 10
- **Evaluation**: Classification report and prediction scatter plot

## Project Structure
├── Model.py # Main script
├── requirements.txt # Reuirements 
└── README.md # Project documentation

Running the Project
`python Model.py`

# This will:

1. Load and split the Iris dataset.
2. Create a preprocessing and modeling pipeline.
3. Perform a grid search to find the best n_neighbors.
4. Evaluate the model on a validation set.
5. Visualize predictions vs actual values.

## Sample Output
```

{'memory': None, 'steps': [('scale', StandardScaler()), ('poly', PolynomialFeatures()), ('model', KNeighborsClassifier())], 
'transform_input': None, 'verbose': False, 'scale': StandardScaler(), 'poly': PolynomialFeatures(), 'model': KNeighborsClassifier(), 
'scale__copy': True, 'scale__with_mean': True, 'scale__with_std': True, 'poly__degree': 2, 'poly__include_bias': True, 'poly__interaction_only': False,
'poly__order': 'C', 'model__algorithm': 'auto', 'model__leaf_size': 30, 'model__metric': 'minkowski', 'model__metric_params': None, 'model__n_jobs': None,
'model__n_neighbors': 5, 'model__p': 2, 'model__weights': 'uniform'}

   mean_fit_time  std_fit_time  mean_score_time  std_score_time  ...  split2_test_score mean_test_score  std_test_score  rank_test_score
0       0.001784  9.544449e-04         0.002116        0.000539  ...            0.96875        0.958333        0.014731                2
1       0.000861  6.506716e-05         0.001432        0.000042  ...            0.93750        0.937500        0.025516                9
2       0.000796  1.655632e-06         0.001383        0.000007  ...            0.96875        0.958333        0.014731                2
3       0.000798  4.495664e-07         0.001397        0.000015  ...            0.96875        0.958333        0.014731                2
4       0.000788  8.064831e-06         0.001381        0.000007  ...            0.96875        0.958333        0.014731                2
5       0.000782  4.720447e-06         0.001384        0.000008  ...            0.96875        0.947917        0.014731                7
6       0.000778  3.128850e-06         0.001379        0.000008  ...            0.96875        0.968750        0.025516                1
7       0.000781  8.195344e-06         0.001387        0.000006  ...            0.96875        0.947917        0.029463                7
8       0.000780  3.612302e-06         0.001393        0.000012  ...            0.96875        0.958333        0.014731                2
9       0.000783  6.779925e-06         0.001405        0.000008  ...            0.96875        0.937500        0.025516                9

[10 rows x 12 columns]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         8
           1       1.00      0.82      0.90        11
           2       0.71      1.00      0.83         5

    accuracy                           0.92        24
   macro avg       0.90      0.94      0.91        24
weighted avg       0.94      0.92      0.92        24

```

## The model achieves ~92% accuracy on the validation set.

## Visualizations

* A scatter plot showing the distribution of the first two features colored by species.
* A scatter plot of predicted vs actual class labels on the validation set.

* 
## Notes

* Although PolynomialFeatures is more commonly used with linear models, it's included here to demonstrate feature expansion and its compatibility with KNN.
* You can modify the pipeline to experiment with other classifiers like Logistic Regression or Decision Trees.

