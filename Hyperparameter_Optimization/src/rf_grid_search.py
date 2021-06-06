import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":

    df = pd.read_csv("../input/mobile_train.csv")

    X = df.drop(columns=['price_range']).values
    y = df['price_range'].values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)

    # Define grid of parameters

    param_grid = {
        "n_estimators" : [100,200,250,300,400,500],
        "max_depth" : [1,2,5,7,11,15],
        "criterion" : ["gini","entropy"]
    }

    # Initialize Grid Search

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=-1,
        cv=5)

    model.fit(X,y)
    print(f"Best score: {model.best_score_}")

    print("Best Params set:")
    best_parameters =  model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
