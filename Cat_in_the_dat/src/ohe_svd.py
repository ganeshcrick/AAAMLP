import pandas as pd

from scipy import sparse
from sklearn import decomposition
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

    # Initialzie Truncated SVD
    # Reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    
    # Fit SVD on Full SParse Train and vali data
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # Initialize Random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train['target'].values)

    # predict on validation data# we need the probability values as we are calculating AUC# we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid['target'].values, valid_preds)

    print(f"Fold:{fold}, AUC:{auc}")

if __name__ == "__main__":
    for i in range(1):
        run(i)

