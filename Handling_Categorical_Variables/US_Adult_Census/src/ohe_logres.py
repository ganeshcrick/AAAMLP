import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):

    df = pd.read_csv("../input/adult_folds.csv")

    # List of numerical cols
    num_cols=['fnlwgt','age','capital.gain','capital.loss','hours.per.week']

    # drop numerical columns
    df = df.drop(num_cols,axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    ## All columns are features except id and Target

    features = [x for x in df.columns if x not in ["kfold","income"]]

    # fill all NaN values with NONE # note that I am converting all columns to "strings" # it doesn’t matter because all are categories

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid = df[df['kfold']==fold].reset_index(drop=True)

    # Initialize one hot encode
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on train plus validation features
    full_data = df[features]
    ohe.fit(full_data)

    # Transform Train and Valid feature data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # Initialize Logistic regression Model

    model = linear_model.LogisticRegression()
    model.fit(x_train,df_train['income'].values)

# Predict Probablity values to calculate AUC
# We will use the probablity of 1's
    valid_preds = model.predict_proba(x_valid)[:,1]

    # ROC AUC score
    auc = metrics.roc_auc_score(df_valid['income'].values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for f in range(1):
        run(f)


