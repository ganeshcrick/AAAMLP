import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics,preprocessing
from tensorflow.keras import layers,optimizers,callbacks,utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model

def create_model(data, catcols):
    """This function returns a compiled tf.keras model for entity embeddings:
    param data: this is a pandas dataframe:
    param catcols: list of categorical column names:
    return: compiled tf.keras model"""

    # init list of inputs for embeddings
    inputs = []

    # init list of outputs for embeddings
    outputs = []

    # loop over all categorical columns
    for c in catcols:
        # num of nuique values in the columns
        num_unique_values = int(data[c].nunique())
        # simple dimension of embedding calculator
        # min size is half of the number of unique values
        # max size is 50. max size depends on the number of unique categories too. 50 is quite sufficient most of the times but if you have millions of unique values, you might need a larger dimension

        embed_dim = int(min(np.ceil((num_unique_values)/2),50))

        # Simple Keras input Layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(num_unique_values + 1, embed_dim,name = c)(inp)

        # 1-d spatial dropout is the standard for embedding layers
        # you can use it in NLP tasks too
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of the embedding, this becomes our output layer for current feature
        out = layers.Reshape(target_shape = (embed_dim,))(out)

        # Add inputs and outputs to respective list
        inputs.append(inp)
        outputs.append(out)
    
    # Concatenate all output layers
    x = layers.Concatenate()(outputs)

    # Add a batchnorm layer
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with dropout.
    # start with 1 or 2 layers only
    x = layers.Dense(300,activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300,activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax and treating it as a two class problem# you can also use sigmoid, then you need to use only one output class
    y = layers.Dense(2, activation="softmax")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)
    # compile the model# we use adam and binary cross entropy.
    # feel free to use something else and see how model behaves
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

    
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

    # create tf.keras model
    model = create_model(df, features)

    # our features are lists of list
    xtrain = [df_train[features].values[:,k] for k in range(len(features))]
    xvalid = [df_valid[features].values[:,k] for k in range(len(features))]

    # Fetch target columns
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # Convert Target columns to Categories,i.e. Binarization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(xtrain,ytrain_cat,validation_data = (xvalid,yvalid_cat),verbose=1,batch_size=1024,epochs=3)

    # generate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]
    # print roc auc score
    print(metrics.roc_auc_score(yvalid, valid_preds))
    
    # clear session to free up some GPU memory
    K.clear_session()

if __name__ == "__main__":
    run(0)
    