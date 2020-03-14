from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



class LOCOModel:
    def __init__(self,data,impFeatures,model):
        self.data = data
        self.impFeatures = impFeatures
        self.model = model
        logging.basicConfig(filename='analysis.log', level=logging.DEBUG)


    '''
    Function to calculate the log loss/ cross entropy
    '''
    def calculateLoss(self,y_true,y_pred,axis):
        return -(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred)).sum(axis=axis)



    '''
    LOCO: Model free variable importance
    '''

    def locoLocal(self):
        k = KFold(n_splits=2)
        importanceMeasure = dict()
        for feature in self.impFeatures:
            importanceMeasure[feature] = list()
        for train_index,test_index in k.split(self.data.iloc[:,1:]):
            train_X,test_X = self.data.iloc[train_index,1:],self.data.iloc[test_index,1:]
            train_y,test_y = self.data.iloc[train_index,0],self.data.iloc[test_index,0]
            for i in range(0,test_X.shape[0]):
                regAlgo = self.model
                regAlgo.fit(train_X.values,list(train_y.values))
                y_pred = regAlgo.predict_proba(test_X.iloc[i].values.reshape(1,-1))
                for feature in self.impFeatures:
                    selective_train_X = train_X.drop([feature],axis=1)
                    selective_test_X = test_X.drop([feature],axis=1)
                    regAlgo2 = self.model
                    regAlgo2.fit(selective_train_X.values,list(train_y.values))
                    y_pred_selective = regAlgo2.predict_proba((selective_test_X.iloc[i].values).reshape(1,-1))
                    importanceMeasure[feature].append(abs(self.calculateLoss(test_y.iloc[i],y_pred_selective,axis=1))-abs(self.calculateLoss(test_y.iloc[i],y_pred,axis=1)))

        return importanceMeasure

    '''
    Calculate accuracy of the model
    '''

    def calculateAccuracy(self,outcome,method):
        train_X, test_X, train_y, test_y = train_test_split(self.data.iloc[:,1:],self.data.iloc[:,0],test_size=0.5)
        k = KFold(n_splits=2)
        accuracy = 0
        for train_index,test_index in k.split(self.data.iloc[:,1:]):
            train_X,test_X = self.data.iloc[train_index,1:],self.data.iloc[test_index,1:]
            train_y,test_y = self.data.iloc[train_index,0],self.data.iloc[test_index,0]
            model = self.model
            model.fit(train_X.values,list(train_y.values))
            y_pred = model.predict(test_X.values)
            accuracy += accuracy_score(list(test_y.values),y_pred)
        logging.debug("Accuracy of predicting {} using {} is {}".format(outcome,method,str(accuracy/2)))
