#!/usr/bin/python3
# inference.py
# Xavier Vasques 13/04/2021


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing
from sklearn.metrics import classification_report



def inference():

    MODEL_DIR = '/Volumes/GL/TECH/IIITH/TA_Session_Material/Material/EEG-letters-main'
    MODEL_PATH_LDA = 'lda.joblib'
    MODEL_PATH_NN = 'nn.joblib'
        
    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    print(MODEL_PATH_LDA)
    clf_lda = load(MODEL_PATH_LDA)
    print("LDA score and classification:")
    prediction_lda = clf_lda.predict(X_test)
    report_lda = classification_report(y_test, prediction_lda)

    print(clf_lda.score(X_test, y_test))
    print('LDA Prediction:', prediction_lda)
    print('LDA Classification Report:', report_lda)

        
    # Run model
    clf_nn = load(MODEL_PATH_NN)
    print("NN score and classification:")
    prediction_nn = clf_nn.predict(X_test)
    report_nn = classification_report(y_test, prediction_nn)


    print(clf_nn.score(X_test, y_test))
    print('NN Prediction:', prediction_nn)
    print('NN Classification Report:', report_nn)

    #Added model run on test data

    clf_dt = load(MODEL_PATH_NN)
    print("DT score and classification:")
    prediction_dt = clf_dt.predict(X_test)
    report_dt = classification_report(y_test, prediction_dt)

    print(clf_dt.score(X_test, y_test))
    print('DT Prediction:', prediction_dt)
    print('DT Classification Report:', report_dt)


    
    
if __name__ == '__main__':
    inference()
