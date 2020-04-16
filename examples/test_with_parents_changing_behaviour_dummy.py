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
from conformalInference.display import Display
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from skfeature.function.wrapper import decision_tree_backward
from skfeature.function.wrapper import decision_tree_forward




'''
Loading data into a python pandas dataframe (set of rows and columns)

'''
data = pd.read_stata("Data/machine_learning_v11.dta")
data.shape
new_variable = list()
for index,row in data.iterrows():
    if row['scale_large4'] == 1 and row['sig'] == 1:
        new_variable.append(3)
    elif row['scale_large4'] == 1 and row['sig'] == 0:
        new_variable.append(2)
    elif row['scale_large4'] == 0 and row['sig'] == 1:
        new_variable.append(1)
    else:
        new_variable.append(0)

data['new_variable'] = pd.Series(new_variable)

data = data[data['citation_index']!='not_reported']
data['citation_index'] = data['citation_index'].astype('int')

data.citation_index.mean()
citation_high = list()

for index,row in data.iterrows():
    if int(row['citation_index']) > 112:
        citation_high.append(1)
    else:
        citation_high.append(0)

data['citation_high'] = pd.Series(citation_high)


'''
Dropping unnecessary rows(not_reported)
'''
data = data[data.parents_changing_behavior_dummy!='not_reported']
data['parents_changing_behavior_dummy'] = data['parents_changing_behavior_dummy'].replace('no_change','no')
#data = data[data.citation_index!='not_reported']
data = data[data.scale_large!=99.0]
data = data[data.scale_large2!=99.0]
data = data[data.scale_large3!=99.0]
data = data[data.scale_large4!=99.0]
data = data[data.scale_large5!=99.0]
data = data[data.scale_large6!=99.0]
data = data[data.scale_large7!=99.0]
data = data.drop(['unit_treatment_household','unit_treatment_site','citation_index','journal_impact_factor'],axis=1)
data.shape

data = data.drop(['paper_name'],axis=1)


'''
Change different columns into categorical types

'''

def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column

data['low_socio_econ'] = changeToCategorical(data['low_socio_econ'])
data['parents_changing_behavior_dummy'] = changeToCategorical(data['parents_changing_behavior_dummy'])
data['outcome_schooling_dummy'] = changeToCategorical(data['outcome_schooling_dummy'])
data['outcome_cognitive_dummy'] = changeToCategorical(data['outcome_cognitive_dummy'])
data['outcome_language_dummy'] = changeToCategorical(data['outcome_language_dummy'])
data['outcome_social_skills_dummy'] = changeToCategorical(data['outcome_social_skills_dummy'])
data['outcome_health_dummy'] = changeToCategorical(data['outcome_health_dummy'])
data['outcome_labour_mkt_dummy'] = changeToCategorical(data['outcome_labour_mkt_dummy'])
data['scale_large2'] = changeToCategorical(data['scale_large2'])
data['scale_large3'] = changeToCategorical(data['scale_large3'])
data['scale_large4'] = changeToCategorical(data['scale_large4'])
#data['citation_index'] = changeToCategorical(data['citation_index'])
data.shape

data.columns
data = data.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,48,49,50,54]]
data.columns
data.to_stata('machine_learning_v11_With_Parents_Dummy.dta')


data.isna().sum()
'''
Selecting one row from each study id  to be used for further analysis

'''
column_names = list(data.columns)
selective_df=pd.DataFrame(columns = column_names)
for index,row in data.groupby(['studyid']):
    random.seed(0)
    random_row = random.randint(0,row.shape[0]-1)
    selective_df = selective_df.append(row.iloc[0])

selective_df.shape

selective_df.columns

selective_df.isna().sum()

#model = DecisionTreeClassifier(random_state=0)
model = LogisticRegression(C=10,solver='liblinear')


'''
Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

'''

Y_index=[54]
featuresDict = dict()
for index in Y_index:
    train_data = selective_df.iloc[:,[index,3,4,12,13,14,15,16,17,18,22,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,52]]
    mrmrModel = MRMR(train_data,[3,4,12,13,14,15,16,17,18,22,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,52],model)
    featuresDict[(data.columns[index])] = mrmrModel.findBestFeaturesMRMR()
mrmrModel.iterateDict(featuresDict)


X = selective_df.iloc[:,[3,4,12,13,14,15,16,17,18,22,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,52]]
Y = selective_df.iloc[:,54]
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
errorIntervals = dict()
for labels in [54]:
    allFeatures = featuresDict[('new_variable')]
    #allFeatures = list(selective_df.columns[[3,4,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,35,36]])
    dataFeatures = [x for x in allFeatures]
    dataFeatures.insert(0,data.columns[labels])
    testing_data = selective_df[dataFeatures]
    locoModel = LOCOModel(testing_data,allFeatures,model)
    locoModel.calculateAccuracy("new_variable","logistic regression")
    importanceMeasureAllFeatures = locoModel.locoLocal()


'''
Performing wilcoxon signed rank test
'''

for key,value in importanceMeasureAllFeatures.items():
    importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])

wilcoxonU = WilcoxonTest(importanceMeasureAllFeatures)
wilcoxonU.test()
wilcoxonU.sort("Results/LogisticRegression/new_variable_scale_significance_combination_with_parents_dummy_mrmr_features_first_row.csv")
df = pd.DataFrame(wilcoxonU.result.items(),columns=["LOCO_results","P-value"])
df['MRMR_results'] = featuresDict['new_variable']

df['Fisher_Score_results'] = pd.Series(feature_list_fisher[0:15])
df['ReliefF_Score_results'] = pd.Series(feature_list_relief[0:15])





'''
Wrapper methods Backward Selection Algorithm

'''
X = X[featuresDict['new_variable']]
Y = Y.astype('int')
backward_algo_indices = decision_tree_backward.decision_tree_backward(X.values,Y.values,n_selected_features=15)

features_backward_algo = list()
for x in backward_algo_indices:
    features_backward_algo.append(X.columns[x])

df['backward_algo_results'] = pd.Series(features_backward_algo)


'''
Forward selection Algorithm
'''
X = X[featuresDict['new_variable']]
Y = Y.astype('int')
forward_algo_indices = decision_tree_forward.decision_tree_forward(X.values,Y.values,n_selected_features=15)

features_forward_algo = list()
for x in forward_algo_indices:
    features_forward_algo.append(X.columns[x])

df['forward_algo_results'] = pd.Series(features_forward_algo)


df.to_csv("Results/LogisticRegression/new_variable_scale_significance_combination_with_parents_dummy_mrmr_features_first_row.csv",index=None)
