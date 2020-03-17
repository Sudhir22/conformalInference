from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import math


class SplitConformal:

    def __init__(self,data_X,data_Y,miscoverage,xNew):
        self.data_X = data_X
        self.data_Y = data_Y
        self.miscoverage = miscoverage
        self.xNew = xNew

    '''
    Function to calculate the log loss/ cross entropy
    '''
    def calculateLoss(self,y_true,y_pred,axis):
        return -(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred)).sum(axis=axis)


    '''
        Split conformal prediction
    '''

    def splitConformalInference(self):
        train_X,test_X,train_Y,test_Y = train_test_split(self.data_X,self.data_Y,test_size=0.5)
        regAlgo = LogisticRegression(C=10)
        regAlgo.fit(train_X,list(train_Y.values))
        y_pred = regAlgo.predict_proba(test_X.values)
        residualsList = list()
        i=0
        labelsList = list(test_Y.values)
        for prediction in y_pred:
            residualsList.append(self.calculateLoss(labelsList[i],prediction,axis=0))
            i+=1
        k = math.ceil(((train_X.shape[0]/2.0)+1)*(1-self.miscoverage))
        sortedlossList = sorted(residualsList)
        d = sortedlossList[k-1]
        return [regAlgo.predict(self.xNew.values.reshape(1,-1))-d, regAlgo.predict(self.xNew.values.reshape(1,-1))+d]
