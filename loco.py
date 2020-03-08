from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np



class LOCOModel:
    def __init__(self,data,impFeatures):
        self.data = data
        self.impFeatures = impFeatures


    '''
    Function to calculate the log loss/ cross entropy
    '''
    def calculateLoss(self,y_true,y_pred,axis):
        return -(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred)).sum(axis=axis)



    '''
    LOCO: Model free variable importance
    '''

    def locoLocal(self):
        train_X, test_X, train_y, test_y = train_test_split(self.data.iloc[:,1:],self.data.iloc[:,0],test_size=0.5)
        importanceMeasure = dict()
        for feature in self.impFeatures:
            importanceMeasure[feature] = list()
        for i in range(0,test_X.shape[0]):
            regAlgo = LogisticRegression(C=10)
            regAlgo.fit(train_X.values,list(train_y.values))
            y_pred = regAlgo.predict_proba((test_X.iloc[i].values).reshape(1,-1))
            for feature in self.impFeatures:
                selective_train_X = train_X.drop([feature],axis=1)
                selective_test_X = test_X.drop([feature],axis=1)
                regAlgo2 = LogisticRegression(C=10)
                regAlgo2.fit(selective_train_X.values,list(train_y.values))
                y_pred_selective = regAlgo2.predict_proba((selective_test_X.iloc[i].values).reshape(1,-1))
                importanceMeasure[feature].append(abs(self.calculateLoss(test_y.iloc[i],y_pred_selective,axis=1))-abs(self.calculateLoss(test_y.iloc[i],y_pred,axis=1)))

        return importanceMeasure
