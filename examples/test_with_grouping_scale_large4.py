import pandas as pd
import random
import math
import numpy as np
import csv
import scipy
from conformalInference.mRMR import MRMR
from conformalInference.loco import LOCOModel
from conformalInference.wilcoxon import WilcoxonTest

'''
Loading row into a python pandas rowframe (set of rows and columns)

'''
data = pd.read_stata("machine_learning_v5.dta")
data.shape


def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column


'''
Grouping row using scale_large4
'''
for index,row in data.groupby(['scale_large4']):

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
    for index,row in row.groupby(['studyid']):
        selective_df = selective_df.append(row.iloc[0])

    print(selective_df.shape)



    '''
    Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

    '''

    Y_index=[35]
    featuresDict = dict()
    for index in Y_index:
        train_data = selective_df.iloc[:,[index,3,11,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,34]]
        mrmrModel = MRMR(train_data,10)
        mrmrModel2 = MRMR(train_data,15)
        featuresDict[(selective_df.columns[index],10)] = mrmrModel.findBestFeatures()
        featuresDict[(selective_df.columns[index],15)] = mrmrModel2.findBestFeatures()
    mrmrModel.iterateDict(featuresDict)


    '''
    Calculating feature importance using LOCO and features found using MRMR

    '''
    importanceMeasureAllFeatures = dict()
    for labels in [35]:
        allFeatures = featuresDict[('sig',10)]
        dataFeatures = [x for x in allFeatures]
        dataFeatures.insert(0,selective_df.columns[labels])
        testing_data = selective_df[dataFeatures]
        locoModel = LOCOModel(testing_data,allFeatures)
        importanceMeasureAllFeatures = locoModel.locoLocal()


    '''
    Performing wilcoxon signed rank test
    '''
    for key,value in importanceMeasureAllFeatures.items():
        importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])

    wilcoxonU = WilcoxonTest(importanceMeasureAllFeatures,[0]*math.ceil(selective_df.shape[0]/2))
    wilcoxonU.test()
    wilcoxonU.sort("sig_importance_group_scale_large4.csv")
