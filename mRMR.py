import pymrmr
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from bartpy.sklearnmodel import SklearnModel
import logging


class MRMR:
    def __init__(self,data,features,model):
        self.data = data
        self.features = features
        self.model = model
        logging.basicConfig(filename="analysis.log",level=logging.DEBUG)



    '''
    Function to iterate over a dictionary
    '''
    def iterateDict(self,dictionary):
        for key,value in dictionary.items():
            logging.debug(key)
            logging.debug(value)


    '''
    Inbuilt MRMR to find top features
    '''
    def findBestFeaturesMRMR(self):
        feature_set = dict()
        '''
        Finding features iteratively from 1 to 15
        '''
        for i in range(0,len(self.features)):
            feature_set[i] = pymrmr.mRMR(self.data,'MID',i+1)

        print(len(feature_set))


        index_feature_set = dict()
        for key,value in feature_set.items():
            index_feature_set[key] = list()
            for v in value:
                index_feature_set[key].append(list(self.data.columns).index(v))

        '''
        Cross-validation to find the best set of features
        '''
        loss = 100
        index = 0
        for i in range(0,15):
            model = self.model
            kf = KFold(n_splits=4)
            total_loss = 0
            for train_index,test_index in kf.split(self.data.iloc[:,1:]):
                train_X,test_X = self.data.iloc[train_index,index_feature_set[i]],self.data.iloc[test_index,index_feature_set[i]]
                train_y,test_y = self.data.iloc[train_index,0],self.data.iloc[test_index,0]
                model.fit(train_X.values,list(train_y.values))
                y_pred = model.predict_proba(test_X.values)
                total_loss += log_loss(list(test_y.values),y_pred)
            if (total_loss/4)<loss:
                loss = total_loss
                index = i


        final_features = list()
        for x in index_feature_set[index]:
            final_features.append(self.data.columns[x])

        return final_features
