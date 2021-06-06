import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from skopt import gp_minimize
from skopt import space

def optimize(params, param_names,x,y):

    # Convert params to dict

    params = dict((zip(param_names, params)))

    # Initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)

    # Initialize STartified k-fold

    kf = model_selection.StratifiedKFold(n_splits=5)

    # Initialize accuracy list
    accuracies = []

    # loop over all folds

    for idx in kf.split(X=x, y=y):
        train_idx,test_idx = idx[0],idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        # fit model for current fold

        model.fit(xtrain,ytrain)
        preds = model.predict(xtest)

        # calulate and append accuracy
        fold_accuracy = metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_accuracy)
    
    return -1 * np.mean(accuracies)


if __name__ == "__main__":

    df = pd.read_csv("../input/mobile_train.csv")

    X = df.drop(columns=['price_range']).values
    y = df['price_range'].values


    # define parameter space
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    # make a list of param names# this has to be same order as the search space# inside the main function
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"]

    # by using functools partial, i am creating a # new function which has same parameters as the 
    # optimize function except for the fact that# only one param, i.e. the "params" parameter is# required. this is how gp_minimize expects the 
    # optimization function to be. you can get rid of this# by reading data inside the optimize function or by
    # defining the optimize function here.
    optimization_function = partial(
        optimize,
        param_names=param_names,
        x=X,
        y=y
        )

    # now we call gp_minimizefrom scikit-optimize# gp_minimize uses bayesian optimization for 
    # minimization of the optimization function.# we need a space of parameters, the function itself,# the number of calls/iterations we want to have
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
        )
        
    # Create best params dict and print

    best_params = dict(zip(param_names,result.x))

    print(best_params)
