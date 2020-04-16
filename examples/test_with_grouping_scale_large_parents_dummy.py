import pandas as pd
import random
import math
import numpy as np
import csv
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from conformalInference.mRMR import MRMR
from conformalInference.loco import LOCOModel
from conformalInference.wilcoxon import WilcoxonTest
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from skfeature.function.wrapper import decision_tree_backward
from skfeature.function.wrapper import decision_tree_forward
from sklearn.linear_model import RidgeClassifier



'''
Loading row into a python pandas rowframe (set of rows and columns)

'''
data = pd.read_stata("machine_learning_v11.dta")
data.shape
data.columns
scale = {0:"small scale",1:"large scale"}
data = data[data['citation_index']!='not_reported']
citation_high = list()

for index,row in data.iterrows():
    if int(row['citation_index']) > 76:
        citation_high.append(1)
    else:
        citation_high.append(0)

data['citation_high'] = pd.Series(citation_high)


#data.describe().to_csv('statistics.csv')
def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column

data = data.dropna()
'''
Grouping row using scale_large4/scale_large3/scale_large2
'''
for index,row in data.groupby(['scale_large4']):
    if index==1:
        '''
        Dropping unnecessary rows(not_reported)
        '''
        #row.describe().to_csv('statistics.csv')
        row = row[row.parents_changing_behavior_dummy!='not_reported']
        row['parents_changing_behavior_dummy'] = row['parents_changing_behavior_dummy'].replace('no_change','no')
        #row = row[row.citation_index!='not_reported']
        row = row.drop(['citation_index'],axis=1)
        row = row[row.scale_large!=99.0]
        row = row.drop(['unit_treatment_household','unit_treatment_site'],axis=1)
        row.shape

        row['low_socio_econ'] = changeToCategorical(row['low_socio_econ'])
        row['parents_changing_behavior_dummy'] = changeToCategorical(row['parents_changing_behavior_dummy'])
        row['outcome_schooling_dummy'] = changeToCategorical(row['outcome_schooling_dummy'])
        row['outcome_cognitive_dummy'] = changeToCategorical(row['outcome_cognitive_dummy'])
        row['outcome_language_dummy'] = changeToCategorical(row['outcome_language_dummy'])
        row['outcome_social_skills_dummy'] = changeToCategorical(row['outcome_social_skills_dummy'])
        row['outcome_health_dummy'] = changeToCategorical(row['outcome_health_dummy'])
        row['outcome_labour_mkt_dummy'] = changeToCategorical(row['outcome_labour_mkt_dummy'])
        row['scale_large2'] = changeToCategorical(row['scale_large2'])
        row['scale_large3'] = changeToCategorical(row['scale_large3'])
        row['scale_large4'] = changeToCategorical(row['scale_large4'])
        #row['citation_index'] = changeToCategorical(row['citation_index'])

        '''
        Selecting one row from each study id  to be used for further analysis

        '''
        column_names = list(row.columns)
        selective_df=pd.DataFrame(columns = column_names)
        for ind,row in row.groupby(['studyid']):
            random.seed(3)
            random_row = random.randint(0,row.shape[0]-1)
            selective_df = selective_df.append(row.iloc[0])

        selective_df.columns
        selective_df.describe().to_csv('Results/SupervisedLearning/first_row_study_id_statistics.csv')
        #model = DecisionTreeClassifier(random_state=0)
        model = LogisticRegression(C=10,solver='liblinear')
        selective_df.sig.unique()

        '''
        Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

        '''

        Y_index=[52]
        featuresDict = dict()
        for ind in Y_index:
            train_data = selective_df.iloc[:,[ind,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,21,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,51,56]]
            mrmrModel = MRMR(train_data,[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,21,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,51,56],model)
            featuresDict[selective_df.columns[ind]] = mrmrModel.findBestFeaturesMRMR()
        mrmrModel.iterateDict(featuresDict)

        X = selective_df.iloc[:,[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,21,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,51,56]]
        Y = selective_df.iloc[:,52]

        '''
        Fisher Score
        '''

        score = fisher_score.fisher_score(X.values,Y.values)
        sorted_score = fisher_score.feature_ranking(score)

        feature_list_fisher = list()
        for x in sorted_score:
            feature_list_fisher.append(X.columns[x])


        '''
        ReliefF algorithm
        '''

        relief_score = reliefF.reliefF(X.values,Y.values)
        sorted_relief_score = reliefF.feature_ranking(relief_score)

        feature_list_relief = list()
        for x in sorted_relief_score:
            feature_list_relief.append(X.columns[x])


        '''
        Calculating feature importance using LOCO and features found using MRMR

        '''
        importanceMeasureAllFeatures = dict()
        for labels in [52]:
            allFeatures = featuresDict['sig']
            #allFeatures = list(selective_df.columns[[3,4,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,31,35]])
            dataFeatures = [x for x in allFeatures]
            dataFeatures.insert(0,selective_df.columns[labels])
            testing_data = selective_df[dataFeatures]
            locoModel = LOCOModel(testing_data,allFeatures,model)
            locoModel.calculateAccuracy("new_variable with scale_large2 {}".format(scale[index]),"logistic regression")
            importanceMeasureAllFeatures = locoModel.locoLocal()


        '''
        Performing wilcoxon signed rank test
        '''
        for key,value in importanceMeasureAllFeatures.items():
            importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])

        wilcoxonU = WilcoxonTest(importanceMeasureAllFeatures)
        wilcoxonU.test()
        wilcoxonU.sort("Results/SupervisedLearning/sig_conditional_on_large_scale_with_parents_dummy_mrmr_features_first_row.csv".format(scale[index]))
        df = pd.DataFrame(wilcoxonU.result.items(),columns=["LOCO_results","P-value"])
        df['MRMR_results'] = featuresDict['sig']

        df['Fisher_Score_results'] = pd.Series(feature_list_fisher[0:15])
        df['ReliefF_Score_results'] = pd.Series(feature_list_relief[0:15])

        '''
        Wrapper methods Backward Selection Algorithm

        '''
        X = X[featuresDict['sig']]
        Y = Y.astype('int')
        backward_algo_indices = decision_tree_backward.decision_tree_backward(X.values,Y.values,n_selected_features=15)

        features_backward_algo = list()
        for x in backward_algo_indices:
            features_backward_algo.append(X.columns[x])

        df['backward_algo_results'] = pd.Series(features_backward_algo)


        '''
        Forward selection Algorithm
        '''
        X = X[featuresDict['sig']]
        Y = Y.astype('int')
        forward_algo_indices = decision_tree_forward.decision_tree_forward(X.values,Y.values,n_selected_features=15)

        features_forward_algo = list()
        for x in forward_algo_indices:
            features_forward_algo.append(X.columns[x])

        df['forward_algo_results'] = pd.Series(features_forward_algo)



        df.to_csv("Results/SupervisedLearning/sig_conditional_on_large_scale_with_parents_dummy_mrmr_features_first_row.csv".format(scale[index]),index=None)

        '''
        Ridge Classifier
        '''
        ridge_df = pd.DataFrame()

        train_X = selective_df[features_backward_algo[0:10]]
        train_y = selective_df[['sig']]

        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])


        model = RidgeClassifier()
        model.fit(train_X,train_y)
        ridge_df['backward_algo_covariates'] = pd.Series(features_backward_algo[0:10])
        ridge_df['backward_algo_coefficents'] = pd.Series(model.coef_[0])



        train_X = selective_df[features_forward_algo[0:10]]
        train_y = selective_df[['sig']]

        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])


        model = RidgeClassifier()
        model.fit(train_X,train_y)
        ridge_df['forward_algo_covariates'] = pd.Series(features_forward_algo[0:10])
        ridge_df['forward_algo_coefficents'] = pd.Series(model.coef_[0])



        train_X = selective_df[list(wilcoxonU.result.keys())[0:3]]
        train_y = selective_df[['sig']]

        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])


        model = RidgeClassifier()
        model.fit(train_X,train_y)
        ridge_df['loco_covariates'] = pd.Series(list(wilcoxonU.result.keys())[0:3])
        ridge_df['loco_coefficents'] = pd.Series(model.coef_[0])


        ridge_df.to_csv("Results/SupervisedLearning/ridge_regression_with_parents_dummy_mrmr_features_first_row.csv".format(scale[index]),index=None)
