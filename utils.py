
import numpy as np 
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from typing import Tuple,List

def train_evaluate_model(pipeline,train:Tuple,val:Tuple,classes:List):
    """
    This functions Trains and evaluates a model that is already initializated.
    model: Initialized model to be train 
    train: Tuple of (Feature,Target) vectors 
    test: Tuple of (Feature,Target) vectors 
    classes: List

    Returns:
    model: Trained model 
    auc: Area under RC curve
    f1_score: weighted f1-score
    conf_matrix: Confussion matrix
    clf_report: Classification report

    """
    X_train,y_train = train 
    X_val,y_val = val

    y_val_bin = label_binarize(y_val,classes=classes)
    
    pipeline.fit(X_train,y_train)
    y_pred_hard = pipeline.predict(X_val)
    y_pred_prob = pipeline.predict_proba(X_val)
    
    auc = np.round(
                    roc_auc_score(
                    y_true=y_val_bin,
                    y_score=y_pred_prob
                    ,average="weighted"
                    , multi_class="ovr")
                   ,3)

    conf_matrix = confusion_matrix(y_true=y_val,y_pred=y_pred_hard) 
    clf_report = classification_report(y_val,y_pred_hard,
                target_names=['bad','moderate','good'],
                output_dict=True)
    
    clf_report = pd.DataFrame().from_dict(clf_report)
    f1_macro,f1_weighted = clf_report.loc['f1-score'][['macro avg','weighted avg']].values.round(2)

    scores = {'auc':auc,'f1_macro':f1_macro,'f1_weighted':f1_weighted} 
    return pipeline,scores,conf_matrix,clf_report