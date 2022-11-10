import numpy as np 
import pandas as pd 
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import RobustScaler

from utils import train_evaluate_model 
import bentoml 
# ----------------DATA PREPARATION--------------------------

df = pd.read_csv('winequalityN.csv')
df.columns = df.columns.str.lower().str.replace(' ','_') 

# Modify Target
bad = (df['quality'] <5) 
moderate = (df['quality'] >=5) & (df['quality'] <7)
good = (df['quality'] >=7)

df.loc[bad,'quality'] = 'bad'
df.loc[moderate,'quality'] = 'moderate'
df.loc[good,'quality'] = 'good'

# Missing values 

cols_with_missing = df.isnull().sum()[(df.isnull().sum() > 0)].index

for col in cols_with_missing: 
    for t in df.type.unique():
        mask = (df.type == t)
        mean = df.loc[mask,col].mean().round(3)
        df.loc[mask,col] = df.loc[mask,col].fillna(mean) 

# Encode Categorical variable and Target

df.type = df['type'].map({'red':0,'white':1})
df.quality= df['quality'].map({'bad':0,'moderate':1,'good':2})

#------------ Modelling -------------------------

# Dataset Split
df_full_train,df_test = train_test_split(df,test_size=0.2,stratify=df.quality,random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.quality.values

df_test = df_test.reset_index(drop=True)
y_test = df_test.quality.values

del df_full_train['quality']
del df_test['quality'] 

#  
features = df_full_train.columns.tolist()
classes = [0,1,2]

train_full_dicts = df_full_train[features].to_dict(orient='records')
test_dicts = df_test[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(train_full_dicts)
X_test = dv.transform(test_dicts)

## Train Model 

params = {
    'train':(X_full_train,y_full_train),
    'val':(X_test,y_test),
    'classes':classes
}

metrics = ['auc','f1_macro','f1_weighted']

best_params = {
    "min_samples_leaf":2,
    "max_depth": 15,
    "n_estimators": 100,
}

steps = [
    ('preprocessing',RobustScaler()),
    ('classifier',RandomForestClassifier(
                **best_params,
                class_weight='balanced',
                random_state=1))
]
pipeline = Pipeline(steps=steps)
pipeline,score,conf_matrix,clf_rep = train_evaluate_model(pipeline,**params)


bento_model = bentoml.sklearn.save_model(
    "wine_quality_randomforest",
    pipeline,
    signatures ={
        "predict":{"batchable":True,"batch_dim":0},
        "predict_proba":{"batchable":True,"batch_dim":0}
    },
    custom_objects={
        "DictVectorizer":dv
    },
    metadata=score
)

print(f"Model saved: {bento_model}")