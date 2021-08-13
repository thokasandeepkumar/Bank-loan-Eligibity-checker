from wsgiref import simple_server
#https://drive.google.com/drive/folders/1hnVu6CjsuULA7A9zLmcv-8siPlA5VF4n

from flask import Flask, request, render_template

import os
import json
from flask_cors import CORS, cross_origin
import glob
from RawDataValidation.rawValidation import Raw_Data_validation
from Data_ingestion.data_ingestion import data_getter
from preprocessingfolder.preprocessing import Preprocessing
import shutil
from application_logging.logger import App_Logger
from problem_utills.preprocessing_utills import utills
from flask import Response
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from preprocessing_utils.express_text_to_number import ExpTextToNum
from preprocessing_utils.get_dependent_and_indepenedent_data import Get_independet_dependent_data
from preprocessing_utils.CreditScoreNormalizer import CreditScoreNormalizer
from preprocessing_utils.CurrentLoanAmountNormalizer import CurrentLoanAmountNormalizer
from preprocessing_utils.MonthDeliquent import MonthDeliquent
from preprocessing_utils.ArrayToDf import ArrayToDf
from preprocessing_utils.HomeOwnSpell import HomeOwnSpell
from preprocessing_utils.PurposeSpell import PurposeSpell
from sklearn.preprocessing import OneHotEncoder
from preprocessing_utils.Knn_Imputer import KNN_Imputer


app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

@app.route("/validate_input_file", methods=['GET'])
@cross_origin()

def validate_input_file():
    file_path = "Input_data"
    raw_validation = Raw_Data_validation(file_path)
    regex = raw_validation.manualRegexCreation()
    raw_validation.validationFileNameRaw(regex)
    return Response("validation successfully!!")


# @app.route("/preprocessing", methods=['GET','POST'])
# @cross_origin()
# def preprocessing():
#     print("started!!!")
#     all_batch_files = glob.glob("Training_Raw_files_validated/Good_Raw/*.csv")
#     if len(all_batch_files) > 0:
#         data_get = data_getter()
#         data = data_get.data_load(all_batch_files[0])
#         prep = Get_independet_dependent_data()
#         file = open("Preprocessing_log/preprocessing_log.txt", 'a+')
#         logger = App_Logger()
#         X,y = prep.get_independent_dependent_data(data)
#         logger.log(file, "data has been segrigated in dependent and independent columns successfully")
#
#
#         pre = Preprocessing()
#
#
#         data = pre.drop_columns(data,['Loan ID','Customer ID'])
#         logger.log(file,"['Loan ID','Customer ID'] has been dropped successfully")
#
#         nul_col = pre.get_null_column_list(data)
#         #message = str("having the null values ", nul_col)
#         logger.log(file, "Null Column has been extracted")
#
#         problem_utils = utills()
#         Cr_score = problem_utils.handle_incorrect_credit_score(data)
#         logger.log(file,"Invalid credit score has been handled successfully")
#         data['Credit Score'] = Cr_score
#         data = pre.replace_col_value(data,'Home Ownership','HaveMortgage','Home Mortgage')
#         logger.log(file,"HaveMortgage has been replaced Have Mortgage")
#         data = pre.replace_col_value(data,'Purpose','other','Other')
#         logger.log(file,"other has been replaced Have Other")
#         data = pre.replace_col_value(data, 'Purpose', 'Take a Trip', 'vacation')
#         logger.log(file,"Take a Trip has been replaced Have vacation")
#         print(data.columns)
#         trans_dataframe = pre.fill_missing_value_KNN_imputer(data)
#         logger.log(file,"fill the mising value using the KNN_Imputer")
#         logger.log(file, "after imputation new Data frame has been created")
#         New_Data = pd.concat([trans_dataframe, data[data.select_dtypes(exclude=np.number).columns.to_list()]], axis=1)
#         logger.log(file,"catogrical columns and numerical columns")
#         highly_corr_col = pre.get_highly_corr_col_list(New_Data)
#         #New_Data = pre.roundOff_Bankruptcies(New_Data)
#         #logger.log(file,"Bankruptcies column has been rounded of")
#         cat_col = pre.get_cat_col_list(New_Data)
#         print(cat_col)
#
#         New_Data.to_csv(problem_utils.saveResult_result())
#         #shutil.move("Training_Raw_files_validated/Good_Raw/credit.csv","Preprocessed_Data")
#
#         return Response("preprocessing successfull!!")

@app.route("/preprocessing_new", methods=['GET','POST'])
@cross_origin()
def preprocessing_new():
    print("started!!!")

    all_batch_files = glob.glob("Training_Raw_files_validated/Good_Raw/*.csv")
    if len(all_batch_files) > 0:
        data_get = data_getter()
        pre = Preprocessing()
        data = data_get.data_load(all_batch_files[0])
        prep = Get_independet_dependent_data()

        X, y = prep.get_independent_dependent_data(data)

        print("independent columns:-",X.columns)


        expTextToNum = ExpTextToNum(data)
        drop_cols = ['Loan ID', 'Customer ID', 'Bankruptcies', 'Tax Liens']
        ##testing the combo
        credit_score_normalizer = CreditScoreNormalizer(data)

        currentScore_normalizer = CurrentLoanAmountNormalizer(X)
        monthly_deliquent = MonthDeliquent(X)
        home_own_spell = HomeOwnSpell(X)
        purpose_Spell = PurposeSpell(X)
        ###For categorical values we need one hot encoding also
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        # KNN Imputation Need to be done here
        #knn_imputer = KNN_Imputer(X)

        # num_col = pre.get_numerical_col(data)


        # ct2 = make_column_transformer((expTextToNum, ['Years in current job']), \
        #                               (credit_score_normalizer, ['Credit Score']), \
        #                               (currentScore_normalizer, ['Current Loan Amount']), \
        #                               (monthly_deliquent,['Months since last delinquent']),\
        #                               (home_own_spell,['Home Ownership']),\
        #                               (purpose_Spell,['Purpose']),\
        #
        #                               (ohe, ['Term', 'Home Ownership', 'Purpose']),\
        #
        #                               ('drop', drop_cols), \
        #                               remainder='passthrough')
        ct2 = make_column_transformer((expTextToNum, ['Years in current job']), \
                                      (credit_score_normalizer, ['Credit Score']), \
                                      (currentScore_normalizer, ['Current Loan Amount']), \
                                      (monthly_deliquent, ['Months since last delinquent']), \
                                      (home_own_spell, ['Home Ownership']), \
                                      #(purpose_Spell,['Purpose'])
                                      ('drop', drop_cols), \
                                      remainder='passthrough')

        pipchk=make_pipeline(ct2,ArrayToDf(X))
        dataframe = pipchk.fit_transform(X)

        purposeSpell_ct = make_column_transformer((PurposeSpell(dataframe), ['Purpose']), remainder='passthrough')

        pipchk =make_pipeline(purposeSpell_ct,ArrayToDf(dataframe))
        dataframe = pipchk.fit_transform(dataframe)
        # dataframe = pd.DataFrame(purposeSpell_ct.fit_transform(dataframe),
        #              columns=['Years in current job', 'Credit Score', 'Current Loan Amount',
        #                       'Months since last delinquent', 'Term', 'Annual Income',
        #                       'Home Ownership', 'Purpose', 'Monthly Debt', 'Years of Credit History',
        #                       'Number of Open Accounts', 'Number of Credit Problems',
        #                       'Current Credit Balance', 'Maximum Open Credit'])



        # Normalized_data = ct2.fit_transform(X)
        #saveResult_result(), index = False
        problem_utils = utills()
        dataframe.to_csv(problem_utils.saveResult_result(),index=False)

        print("Dataframe")
        print(dataframe)



        #trans_dataframe = pre.fill_missing_value_KNN_imputer(data1)
        #trans_dataframe.to_csv(problem_utils.saveResult_result())
        print("///////////////////KNN Data Transform Data frame///////////////////")
        #print(trans_dataframe)




        # print(dataframe.columns)
        # print(Normalized_data)
        #shutil.move("Training_Raw_files_validated/Good_Raw/credit.csv", "Preprocessed_Data")
        print("done")



        return Response("preprocessing successfull!!")

@app.route("/test", methods=['GET'])
@cross_origin()
def test():
    return "success"


if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host="127.0.0.1")
    #app.run(debug=True)



