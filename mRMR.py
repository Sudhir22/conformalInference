import pymrmr
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import logging


class MRMR:
    def __init__(self,data,features,model):
        self.data = data
        self.features = features
        self.model = model
        logging.basicConfig(filename="analysis.log",level=logging.DEBUG)



    '''
    MrMR algoithm to find the top features

    '''

    def findBestFeatures(self):
        feature_set = dict()
        mi_outcome_score = list(mutual_info_classif(self.data.iloc[:,1:].values,list(self.data.iloc[:,0].values)))
        for i in range(0,len(self.features)):
            if i==0:
                feature_set[i] = list()
                feature_set[i].append(mi_outcome_score.index(max(mi_outcome_score))+1)
            else:
                feature_set[i] = feature_set[i-1]
                max_value = 0
                index = 0
                for j in range(0,len(self.features)):
                    sum_of_mutual_score = 0
                    if (j+1) not in feature_set[i-1]:
                        for y in feature_set[i-1]:
                            sum_of_mutual_score = sum_of_mutual_score + mutual_info_score(self.data.iloc[:,j+1].values,list(self.data.iloc[:,y].values))
                        if i>1:
                            mutual_info = mi_outcome_score[j] - (sum_of_mutual_score)/(i-1)
                        else:
                            mutual_info = mi_outcome_score[j] - (sum_of_mutual_score)
                        if mutual_info>max_value:
                            max_value = mutual_info
                            index = j+1
                if index!=0:
                    feature_set[i].append(index)


        loss = 100
        index = 0
        for i in range(0,len(self.features)):
            model = self.model
            kf = KFold(n_splits=5)
            total_loss = 0
            for train_index,test_index in kf.split(self.data.iloc[:,1:]):
                train_X,test_X = self.data.iloc[train_index,1:],self.data.iloc[test_index,1:]
                train_y,test_y = self.data.iloc[train_index,0],self.data.iloc[test_index,0]
                model.fit(train_X.values,list(train_y.values))
                y_pred = model.predict_proba(test_X.values)
                total_loss += log_loss(list(test_y.values),y_pred)
            if (total_loss/5)<loss:
                loss = total_loss
                index = i


        final_features = list()
        for x in feature_set[index]:
            final_features.append(self.data.columns[x])

        return final_features


    '''
    Function to iterate over a dictionary

    '''
    def iterateDict(self,dictionary):
        for key,value in dictionary.items():
            logging.debug(key)
            logging.debug(value)
