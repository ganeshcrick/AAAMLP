import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    
    # data_path =  r'C:\GN\Projects\Datasets\AAAMLP_datasets/'
    # output_path = r'C:\GN\Projects\Datasets\AAAMLP_outputs/MNIST'
    
    df = pd.read_csv("../input/mnist_train_folds.csv")

    # Train data is where kfold is not equal tp provide fold. Also reet index
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    x_train = df_train.drop(columns=['label']).values
    y_train = df_train['label'].values

    x_valid = df_valid.drop(columns=['label']).values
    y_valid = df_valid['label'].values

    # Simple Decision tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)

    # Accuracy
    accuracy =  metrics.accuracy_score(y_valid,preds)
    print(f"Fold={fold}, Accuracy = {accuracy}")

    # save the model
    joblib.dump(clf,f"../models/dt_{fold}.bin")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)