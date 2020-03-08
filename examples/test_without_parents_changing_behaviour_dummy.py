import pandas as pd
import random
import math
import numpy as np
import csv
import scipy
from conformalInference.mRMR import MRMR
from conformalInference.loco import LOCOModel

'''
Loading data into a python pandas dataframe (set of rows and columns)

'''
data = pd.read_stata("machine_learning_v5.dta")
data.shape

'''
Dropping unnecessary rows(not_reported)
'''
data = data.drop(['parents_changing_behavior_dummy'],axis=1)
data = data[data.citation_index!='not_reported']
data = data[data.scale_large!=99.0]
data = data.drop(['unit_treatment_household','unit_treatment_site'],axis=1)
data.shape



'''
Change different columns into categorical types

'''

def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column

data['low_socio_econ'] = changeToCategorical(data['low_socio_econ'])
data['outcome_schooling_dummy'] = changeToCategorical(data['outcome_schooling_dummy'])
data['outcome_cognitive_dummy'] = changeToCategorical(data['outcome_cognitive_dummy'])
data['outcome_language_dummy'] = changeToCategorical(data['outcome_language_dummy'])
data['outcome_social_skills_dummy'] = changeToCategorical(data['outcome_social_skills_dummy'])
data['outcome_health_dummy'] = changeToCategorical(data['outcome_health_dummy'])
data['outcome_labour_mkt_dummy'] = changeToCategorical(data['outcome_labour_mkt_dummy'])
data['scale_large2'] = changeToCategorical(data['scale_large2'])
data['scale_large3'] = changeToCategorical(data['scale_large3'])
data['scale_large4'] = changeToCategorical(data['scale_large4'])
data['citation_index'] = changeToCategorical(data['citation_index'])


'''
Selecting one row from each study id  to be used for further analysis

'''
column_names = list(data.columns)
selective_df=pd.DataFrame(columns = column_names)
for index,row in data.groupby(['studyid']):
    selective_df = selective_df.append(row.iloc[0])

selective_df.shape

selective_df.columns





'''
Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

'''

Y_index=[31,32,33]
featuresDict = dict()
for index in Y_index:
    train_data = selective_df.iloc[:,[index,3,11,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,34,35]]
    mrmrModel = MRMR(train_data,10)
    mrmrModel2 = MRMR(train_data,15)
    featuresDict[(data.columns[index],10)] = mrmrModel.findBestFeatures()
    featuresDict[(data.columns[index],15)] = mrmrModel2.findBestFeatures()
mrmrModel.iterateDict(featuresDict)



'''
Calculating feature importance using LOCO and features found using MRMR

'''
importanceMeasureAllFeatures = dict()
for labels in [33]:
    allFeatures = featuresDict[('scale_large4',10)]
    dataFeatures = [x for x in allFeatures]
    dataFeatures.insert(0,data.columns[labels])
    testing_data = selective_df[dataFeatures]
    locoModel = LOCOModel(testing_data,allFeatures)
    importanceMeasureAllFeatures = locoModel.locoLocal()


'''
Performing wilcoxon signed rank test 
'''

wilcoxonU = dict()

for key,value in importanceMeasureAllFeatures.items():
    importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])
    wilcoxonU[key] = scipy.stats.wilcoxon(importanceMeasureAllFeatures[key],[0]*143)

print(wilcoxonU)
