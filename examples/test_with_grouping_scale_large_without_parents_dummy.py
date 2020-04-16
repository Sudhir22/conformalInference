import pandas as pd
import random
import math
import numpy as np
import csv
import scipy
from sklearn.linear_model import LogisticRegression
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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from regressors import stats
import statsmodels.api as sm


'''
Loading row into a python pandas rowframe (set of rows and columns)

'''
data = pd.read_stata("Data/machine_learning_v17.dta")
data.columns

data.cohens_d.min()

data = data[data['citation_index']!='not_reported']
data['citation_index'] = data['citation_index'].astype('int')

data['citation_index'].mean()


features_for_analysis = ['low_socio_econ','only_govt_treatment_deliverer','only_ngo_treatment_deliverer',
       'only_teacher_treatment_deliverer','only_res_treatment_deliverer', 'only_medical_treatment_deliverer',
       'outcome_measured_survey','only_parents_treated','only_kids_treated', 'only_teachers_treated', 'cash_c', 'developed',
       'treatment_fin_carec','younger_child','time_post_intervention_short','econ_journal', 'only_gov_org_intervention','only_ngo_org_intervention',
       'only_researcher_org_intervention','govt_ngo_org_intervention', 'with_pvt_org_intervention','only_gov_data_collection', 'only_ngo_data_collection',
       'only_researcher_data_collection', 'govt_ngo_data_collection','with_pvt_data_collection', 'medical_data_collection', 'duration_short',
       'auth4', 'working_paper_dum','single_treatment','citation_high']

'''
Creating a new categorical variable citation_high
'''
citation_high = list()
for index,row in data.iterrows():
    if int(row['citation_index']) > 112:
        citation_high.append(1)
    else:
        citation_high.append(0)

data['citation_high'] = pd.Series(citation_high)

'''
Function to change columns into categorical type
'''
def changeToCategorical(column):
    column = pd.Categorical(column)
    column = column.codes
    return column



'''
Grouping row using scale_large4
'''
for index,row in data.groupby(['scale_large4']):
    if index==1:
        '''
        Data cleaning

        '''

        '''
        Dropping unnecessary rows(not_reported)
        '''
        row = row.drop(['parents_changing_behavior_dummy'],axis=1)
        row = row.drop(['citation_index'],axis=1)
        row = row.drop(['unit_treatment_household','unit_treatment_site'],axis=1)
        row['low_socio_econ'] = changeToCategorical(row['low_socio_econ'])
        row['outcome_schooling_dummy'] = changeToCategorical(row['outcome_schooling_dummy'])
        row['outcome_cognitive_dummy'] = changeToCategorical(row['outcome_cognitive_dummy'])
        row['outcome_language_dummy'] = changeToCategorical(row['outcome_language_dummy'])
        row['outcome_social_skills_dummy'] = changeToCategorical(row['outcome_social_skills_dummy'])
        row['outcome_health_dummy'] = changeToCategorical(row['outcome_health_dummy'])
        row['outcome_labour_mkt_dummy'] = changeToCategorical(row['outcome_labour_mkt_dummy'])

        row.shape

        '''
        Selecting one row from each study id to be used for further analysis

        '''

        row = row[(row.outcome_cognitive_dummy==1) | (row.outcome_language_dummy==1) | (row.outcome_schooling_dummy==1)]

        row = row[(row.outcome_health_dummy==0) & (row.outcome_social_skills_dummy==0)]





        row.shape





        column_names = list(row.columns)
        selective_df=pd.DataFrame(columns = column_names)
        random.seed(1)
        for ind,r in row.groupby(['studyid']):
            random_row = random.randint(0,r.shape[0]-1)
            '''
            Selecting random or first row from each study_id
            '''
            #selective_df = selective_df.append(r.iloc[random_row])
            selective_df = selective_df.append(r.iloc[0])


        #selective_df.to_csv('LatestResults/SupervisedLearning/Sample_scale_large4_sig_outcome_first_row.csv')


        features_for_analysis_df = ['low_socio_econ','only_govt_treatment_deliverer','only_ngo_treatment_deliverer',
               'only_teacher_treatment_deliverer','only_res_treatment_deliverer', 'only_medical_treatment_deliverer',
               'outcome_measured_survey','only_parents_treated','only_kids_treated', 'only_teachers_treated', 'cash_c', 'developed',
               'treatment_fin_carec','younger_child','time_post_intervention_short','econ_journal', 'only_gov_org_intervention','only_ngo_org_intervention',
               'only_researcher_org_intervention','govt_ngo_org_intervention', 'with_pvt_org_intervention','only_gov_data_collection', 'only_ngo_data_collection',
               'only_researcher_data_collection', 'govt_ngo_data_collection','with_pvt_data_collection', 'medical_data_collection', 'duration_short',
               'auth4', 'working_paper_dum','single_treatment','citation_high','sig']

        #selective_df[features_for_analysis_df].to_csv("health_social.csv")
        #selective_df.describe().to_csv('FinalResult/SupervisedLearning/first_row_study_id_statistics.csv')
        analysis_feat = ['econ_journal','auth4','working_paper_dum','citation_high','only_parents_treated','cash_c',
                            'low_socio_econ','only_medical_treatment_deliverer','only_ngo_treatment_deliverer','only_researcher_org_intervention']

        statistics_features = pd.DataFrame(columns=analysis_feat)
        for feat in analysis_feat:
            temp_list = list()
            for a,b in selective_df.groupby([feat,'sig']):
                temp_list.append(b.shape[0]/selective_df.shape[0]*1.0)

            statistics_features[feat] = pd.Series(temp_list)


        #statistics_features.to_csv('FinalResult/SupervisedLearning/features_statistics_conditioned_on_health_and_social_skills_inside_sample.csv')


        '''
        Analysis

        '''

        model = LogisticRegression(C=10,solver='liblinear')

        '''
        Calculating the top 'k' features for each outcome variable using MrMR (k={10,15})

        '''

        featuresDict = dict()
        data_features = features_for_analysis
        data_features.insert(0,'sig')
        train_data = selective_df[data_features]
        train_data = train_data.dropna()
        mrmrModel = MRMR(train_data,features_for_analysis,model)
        featuresDict['sig'] = mrmrModel.findBestFeaturesMRMR()
        mrmrModel.iterateDict(featuresDict)

        featuresDict['sig']

        '''
        Selecting features for Fisher/ReliefF based on supported format
        '''

        temp_X = selective_df[features_for_analysis]
        temp_X = temp_X.dropna()
        X = temp_X.iloc[:,1:]
        Y = temp_X.iloc[:,0]

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

        featuresDict['sig']

        '''
        Calculating feature importance using LOCO and features found using MRMR

        '''
        importanceMeasureAllFeatures = dict()
        allFeatures = featuresDict['sig']
        dataFeatures = [x for x in allFeatures]
        dataFeatures.insert(0,'sig')
        testing_data = selective_df[dataFeatures]
        testing_data = testing_data.dropna()
        locoModel = LOCOModel(testing_data,allFeatures,model)
        locoModel.calculateAccuracy("sig with scale_large4","logistic regression")
        importanceMeasureAllFeatures = locoModel.locoLocal()


        '''
        Performing wilcoxon signed rank test
        '''
        for key,value in importanceMeasureAllFeatures.items():
            importanceMeasureAllFeatures[key] = np.concatenate(importanceMeasureAllFeatures[key])

        wilcoxonU = WilcoxonTest(importanceMeasureAllFeatures)
        wilcoxonU.test()
        wilcoxonU.sort("sig_conditional_on_large_scale_without_parents_dummy_mrmr_features_first_row_conditioned_on_cognitive_language_schooling.csv")
        df = pd.DataFrame(wilcoxonU.result.items(),columns=["LOCO_results","P-value"])
        df['MRMR_results'] = featuresDict['sig']
        df['Fisher_Score_results'] = pd.Series(feature_list_fisher[0:15])
        df['ReliefF_Score_results'] = pd.Series(feature_list_relief[0:15])


        '''
        Wrapper methods Backward Selection Algorithm

        '''
        X = X[featuresDict['sig']]
        Y = Y.astype('int')
        X.to_csv('health_X.csv')
        Y.to_csv('health_Y.csv')
        backward_algo_indices = decision_tree_backward.decision_tree_backward(X.values,Y.values,n_selected_features=10)

        features_backward_algo = list()
        for x in backward_algo_indices:
            features_backward_algo.append(X.columns[x])

        df['backward_algo_results'] = pd.Series(features_backward_algo)


        '''
        Forward selection Algorithm
        '''
        X = X[featuresDict['sig']]
        Y = Y.astype('int')
        forward_algo_indices = decision_tree_forward.decision_tree_forward(X.values,Y.values,n_selected_features=10)

        features_forward_algo = list()
        for x in forward_algo_indices:
            features_forward_algo.append(X.columns[x])

        df['forward_algo_results'] = pd.Series(features_forward_algo)



        df.to_csv("sig_conditional_on_large_scale_without_parents_dummy_mrmr_features_first_row_conditioned_on_cognitive_language_schooling.csv",index=None)


        '''
        Ridge Classifier/Logistic Regression
        '''
        ridge_df = pd.DataFrame()

        train_X = selective_df[features_backward_algo[0:15]]
        train_y = selective_df[['sig']]
        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])


        train_X = train_X.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        train_y = train_y.astype('int')

        parameters = {'solver':('lbfgs','liblinear'),'C':[10,20,30,40,50,60,70]}
        #parameters = {'solver':('svd','cholesky','lsqr'),'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.01,0.05,0.001]}
        accuracy_metric = 0
        final_model = None
        for x in parameters['solver']:
            for y in parameters['C']:
                model = LogisticRegression(C=y,solver=x)
                #model = RidgeClassifier(alpha=y,solver=x)
                model.fit(train_X.values,train_y.values)
                if max(cross_val_score(model,train_X.values,train_y.values,scoring='accuracy'))>accuracy_metric:
                    y_pred = cross_val_predict(model,train_X.values,train_y.values,cv=4)
                    accuracy_metric = max(cross_val_score(model,train_X.values,train_y.values,scoring='accuracy'))
                    final_model = model
        print("Confusion Matrix for Backward Algo\n")
        print(confusion_matrix(train_y.values,y_pred))

        print("Accuracy of Backward Algo is {} \n \n".format(str(accuracy_metric)))

        ridge_df['backward_algo_covariates'] = pd.Series(features_backward_algo[0:15])
        ridge_df['backward_algo_coefficents'] = pd.Series(final_model.coef_[0])


        train_X = selective_df[features_forward_algo[0:15]]
        train_y = selective_df[['sig']]

        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])

        train_X = train_X.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        train_y = train_y.astype('int')
        parameters = {'solver':('lbfgs','liblinear'),'C':[10,20,30,40,50,60,70,80,100]}
        #parameters = {'solver':('svd','cholesky','lsqr'),'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.01,0.05,0.001]}
        accuracy_metric = 0
        final_model = None

        for x in parameters['solver']:
            for y in parameters['C']:
                model = LogisticRegression(C=y,solver=x)
                #model = RidgeClassifier(alpha=y,solver=x)
                model.fit(train_X.values,train_y.values)
                if max(cross_val_score(model,train_X.values,train_y.values))>accuracy_metric:
                    y_pred = cross_val_predict(model,train_X.values,train_y.values,cv=4)
                    accuracy_metric = max(cross_val_score(model,train_X.values,train_y.values))
                    final_model = model
        print("Confusion Matrix for Forward Algo\n")
        print(confusion_matrix(train_y.values,y_pred))
        print("Accuracy of Forward Algo is {}\n \n".format(str(accuracy_metric)))


        ridge_df['forward_algo_covariates'] = pd.Series(features_forward_algo[0:15])
        ridge_df['forward_algo_coefficents'] = pd.Series(final_model.coef_[0])




        train_X = selective_df[list(wilcoxonU.result.keys())[0:3]]
        train_y = selective_df[['sig']]

        for column in train_X.columns:
            train_X[column] = changeToCategorical(train_X[column])


        train_X = train_X.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        train_y = train_y.astype('int')

        parameters = {'solver':('lbfgs','liblinear'),'C':[10,20,30,40,50,60,70,80,100]}
        #parameters = {'solver':('svd','cholesky','lsqr'),'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.01,0.05,0.001]}
        accuracy_metric = 0
        final_model = None
        for x in parameters['solver']:
            for y in parameters['C']:
                model = LogisticRegression(C=y,solver=x)
                #model = RidgeClassifier(alpha=y,solver=x)
                model.fit(train_X.values,train_y.values)
                if max(cross_val_score(model,train_X.values,train_y.values))>accuracy_metric:
                    y_pred = cross_val_predict(model,train_X.values,train_y.values,cv=4)
                    accuracy_metric = max(cross_val_score(model,train_X.values,train_y.values))
                    final_model = model
        print("Confusion Matrix for LOCO\n")
        print(confusion_matrix(train_y.values,y_pred))
        print("Accuracy of LOCO is {}".format(str(accuracy_metric)))
        ridge_df['loco_covariates'] = pd.Series(list(wilcoxonU.result.keys())[0:3])
        ridge_df['loco_coefficents'] = pd.Series(final_model.coef_[0])


        ridge_df.to_csv("logistic_regression_for_sig_outcome_large_scale_without_parents_dummy_mrmr_features_first_row_cognitive_language.csv",index=None)
