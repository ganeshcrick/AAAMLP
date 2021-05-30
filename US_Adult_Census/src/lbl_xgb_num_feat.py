import itertools
import pandas as pd
import xgboost as xgb

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df,cat_cols):
    # this will create all 2-combinations of values# in this list# for example:# list(itertools.combinations([1,2,3], 2)) will return# [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols,2))

    for c1,c2 in combi:
        df.loc[:,c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df


def run(fold):

    df = pd.read_csv("../input/adult_folds.csv")

    # List of numerical cols
    num_cols=['fnlwgt','age','capital.gain','capital.loss','hours.per.week']

    # drop numerical columns
    # df = df.drop(num_cols,axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)


    # Categorical Columns for feature engineering
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ["kfold","income"]]

    # Add New Features
    df = feature_engineering(df,cat_cols)

    ## All columns are features except id and Target

    features = [x for x in df.columns if x not in ["kfold","income"]]

    # fill all NaN values with NONE # note that I am converting all columns to "strings" # it doesn’t matter because all are categories

    for col in features:
        if col not in num_cols:# Do not encode numerical columns
            df[col] = df[col].astype(str).fillna("NONE")

    # Label encode Features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df[col] = lbl.transform(df[col])

    df_train = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid = df[df['kfold']==fold].reset_index(drop=True)

    
    # Transform Train and Valid feature data
    x_train = df_train[features]
    x_valid = df_valid[features]

    # Initialize Random Forest Model

    model = xgb.XGBClassifier(n_jobs=-1)
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


