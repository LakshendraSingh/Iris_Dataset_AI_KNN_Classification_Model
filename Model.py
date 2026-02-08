from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.2, random_state=42)

for i, target_name in enumerate(iris.target_names):
    plt.scatter(
        iris.data[iris.target == i, 0],
        iris.data[iris.target == i, 1],
        label=target_name
    )
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(loc='upper left')
plt.show()

pipe = Pipeline([
    ("scale",  StandardScaler()),
    ("poly", PolynomialFeatures()),
    ("model", knn())
])
print(pipe.get_params())
mod = GridSearchCV(estimator=pipe,
                   param_grid = {"model__n_neighbors" : [1,2,3,4,5,6,7,8,9,10]},
                   cv=3
)
mod.fit(X_train,y_train)
print(pd.DataFrame(mod.cv_results_))
p = mod.predict(X_test)
print(f"Best Number of Neighbors (K): {mod.best_params_['model__n_neighbors']}")
print(classification_report(y_test, p)) 
plt.scatter(p,y_test)
plt.legend()
plt.title("predicted and actual data on Validation")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, p, display_labels=iris.target_names, cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Actual vs Predicted")
plt.grid(False)
plt.show()

