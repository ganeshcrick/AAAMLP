import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv("../input/cat_train_folds.csv")

    ## All columns are features except id and Target

    features = [x for x in df.columns if x not in ["id","target","kfold"]]

    # fill all NaN values with NONE # note that I am converting all columns to "strings" # it doesn’t matter because all are categories

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    # Label encode Features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])

    df_train = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid = df[df['kfold']==fold].reset_index(drop=True)

    
    # Transform Train and Valid feature data
    x_train = df_train[features]
    x_valid = df_valid[features]

    # Initialize Random Forest Model

    model = ensemble.RandomForestClassifier()
    model.fit(x_train,df_train['target'].values)

# Predict Probablity values to calculate AUC
# We will use the probablity of 1's
    valid_preds = model.predict_proba(x_valid)[:,1]

    # ROC AUC score
    auc = metrics.roc_auc_score(df_valid['target'].values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for f in range(5):
        run(f)


