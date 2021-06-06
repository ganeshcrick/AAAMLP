import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params,x,y):

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
    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth",1,15,1)),
        "n_estimators": scope.int(hp.quniform("n_estimators",100,1500,1)),
        "criterion": hp.choice("criterion",["gini","entropy"]),
        "max_features": hp.uniform("max_features",0,1)
    }

    # by using functools partial, i am creating a # new function which has same parameters as the 
    # optimize function except for the fact that# only one param, i.e. the "params" parameter is# required. this is how gp_minimize expects the 
    # optimization function to be. you can get rid of this# by reading data inside the optimize function or by
    # defining the optimize function here.
    optimization_function = partial(
        optimize,
        x=X,
        y=y
        )
    ## Initialize Trials to keep logging info
    trials = Trials()

    # now we call gp_minimizefrom scikit-optimize# gp_minimize uses bayesian optimization for 
    # minimization of the optimization function.# we need a space of parameters, the function itself,# the number of calls/iterations we want to have
    hopt = fmin(
        fn = optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=3,
        trials=trials
        )
        
    # Create best params dict and print

    print(hopt)
