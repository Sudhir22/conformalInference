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

'''
Loading row into a python pandas rowframe (set of rows and columns)

'''
data = pd.read_stata("machine_learning_v6.dta")
data.columns

scale = {0:"small scale",1:"large scale"}

def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column


'''
Grouping row using scale_large4/scale_large3/scale_large2
'''
for index,row in data.groupby(['scale_large2']):

    '''
    Dropping unnecessary rows(not_reported)
    '''
    row = row.drop(['parents_changing_behavior_dummy'],axis=1)
    row = row[row.citation_index!='not_reported']
    row = row[row.scale_large!=99.0]
    row = row.drop(['unit_treatment_household','unit_treatment_site'],axis=1)
    row.shape

    row['low_socio_econ'] = changeToCategorical(row['low_socio_econ'])
    row['outcome_schooling_dummy'] = changeToCategorical(row['outcome_schooling_dummy'])
    row['outcome_cognitive_dummy'] = changeToCategorical(row['outcome_cognitive_dummy'])
    row['outcome_language_dummy'] = changeToCategorical(row['outcome_language_dummy'])
    row['outcome_social_skills_dummy'] = changeToCategorical(row['outcome_social_skills_dummy'])
    row['outcome_health_dummy'] = changeToCategorical(row['outcome_health_dummy'])
    row['outcome_labour_mkt_dummy'] = changeToCategorical(row['outcome_labour_mkt_dummy'])
    row['scale_large2'] = changeToCategorical(row['scale_large2'])
    row['scale_large3'] = changeToCategorical(row['scale_large3'])
    row['scale_large4'] = changeToCategorical(row['scale_large4'])
    row['citation_index'] = changeToCategorical(row['citation_index'])

    '''
    Selecting one row from each study id  to be used for further analysis

    '''
    column_names = list(row.columns)
    selective_df=pd.DataFrame(columns = column_names)
    for ind,row in row.groupby(['studyid']):
        selective_df = selective_df.append(row.iloc[0])

    selective_df.shape

    #model = DecisionTreeClassifier(random_state=0)
    model = LogisticRegression(C=10,solver='liblinear')

    '''
    Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

    '''

    Y_index=[35]
    featuresDict = dict()
    for ind in Y_index:
        train_data = selective_df.iloc[:,[ind,3,4,5,6,7,8,9,11,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,34]]
        mrmrModel = MRMR(train_data,[3,4,5,6,7,8,9,11,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,34],model)
        featuresDict[selective_df.columns[ind]] = mrmrModel.findBestFeaturesMRMR()
    mrmrModel.iterateDict(featuresDict)


    '''
    Calculating feature importance using LOCO and features found using MRMR

    '''
    importanceMeasureAllFeatures = dict()
    for labels in [35]:
        allFeatures = featuresDict['sig']
        #allFeatures = list(selective_df.columns[[3,11,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,34]])
        dataFeatures = [x for x in allFeatures]
        dataFeatures.insert(0,selective_df.columns[labels])
        testing_data = selective_df[dataFeatures]
        locoModel = LOCOModel(testing_data,allFeatures,model)
        locoModel.calculateAccuracy("sig with scale_large2 {}".format(scale[index]),"logistic regression")
        importanceMeasureAllFeatures = locoModel.locoLocal()


    '''
    Performing wilcoxon signed rank test
    '''
    for key,value in importanceMeasureAllFeatures.items():
        importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])

    wilcoxonU = WilcoxonTest(importanceMeasureAllFeatures)
    wilcoxonU.test()
    wilcoxonU.sort("Results/LogisticRegression/sig_importance_group_scale_large2_{}_without_parents_dummy_mrmr_features.csv".format(scale[index]))
    df = pd.DataFrame(wilcoxonU.result.items(),columns=["LOCO_results","P-value"])
    df['MRMR_results'] = featuresDict['sig']

    df.to_csv("Results/LogisticRegression/sig_importance_group_scale_large2_{}_without_parents_dummy_mrmr_features.csv".format(scale[index]),index=None)
