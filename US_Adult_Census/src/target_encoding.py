import copy
import pandas as pd
import xgboost as xgb

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def mean_target_encoding(data):

    df = copy.deepcopy(data)

    num_cols = ['fnlwgt','age','capital.gain','capital.loss','hours.per.week']

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
        if col not in num_cols:# Do not encode numerical columns
            df[col] = df[col].astype(str).fillna("NONE")

    # Label encode Features
    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df[col] = lbl.transform(df[col])

    # list to store 5 validation dataframes
    encoded_dfs = []

    # Go over all folds

    for fold in range(5):
        # fetch tarin and valid data
        df_train = df[df['kfold']!= fold].reset_index(drop=True)
        df_valid = df[df['kfold']== fold].reset_index(drop=True)

        # for all categorical columns
        for column in features:
            # create dict of category:mean target
            mapping_dict = dict(df_train.groupby(column)['income'].mean())
            df_valid.loc[:,column + "_enc"] = df_valid[column].map(mapping_dict)
        encoded_dfs.append(df_valid)
    
    encoded_df = pd.concat(encoded_dfs,axis=0)
    return encoded_df




def run(df,fold):

    df_train = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid = df[df['kfold']==fold].reset_index(drop=True)

    features = [x for x in df.columns if x not in ["kfold","income"]]
    
    # Transform Train and Valid feature data
    x_train = df_train[features]
    x_valid = df_valid[features]

    # Initialize Random Forest Model

    model = xgb.XGBClassifier(n_jobs=-1,max_depth=7)
    model.fit(x_train,df_train['income'].values)

# Predict Probablity values to calculate AUC
# We will use the probablity of 1's
    valid_preds = model.predict_proba(x_valid)[:,1]

    # ROC AUC score
    auc = metrics.roc_auc_score(df_valid['income'].values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    
    df = pd.read_csv("../input/adult_folds.csv")
    df = mean_target_encoding(df)
    for f in range(1):
        run(df,f)


