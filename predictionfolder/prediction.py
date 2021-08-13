from preprocessingfolder import preprocessingfile
import pandas as pd
import numpy as np
import joblib
import pickle
from Data_ingestion import data_ingestion
from pandas_profiling import ProfileReport

class Fraud_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.Fraud_preprocess()

        set0 = instance2.initialize_columns(data)
        set1 = instance2.drop_columns(set0)
        df = set1[['customer']]
        set2 = set1.drop(['customer'],axis=1)
        new_data = instance2.obj_to_cat(set2)
        # ============  pandas-prof. report ============================
        new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
        # ==============================================================
        model_Fraud = joblib.load('pickle_files/Fraud_new_model.pkl')
        result = model_Fraud.predict(new_data)

        df['output'] = result
        df['output'] = np.where(df['output'] == 0, "Trusted", "Fraud")
        return df
# ===================================================================================================
class LA_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.LA_preprocess()

        set0 = instance2.initialize_columns(data)
        final_data = set0[['ID']]

        set1=instance2.drop_columns(set0)
        new_data = instance2.encoder(set1)
        # ============  pandas-prof. report ============================
        new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
        # ==============================================================
        ss_LA = pickle.load(open('pickle_files/LA_Std_scaler.pkl', 'rb'))
        model_LA=pickle.load(open('pickle_files/DTModel-1.pkl', 'rb'))

        ss_result = ss_LA.transform(new_data)
        result = model_LA.predict(ss_result)

        new_data['output'] = result
        new_data['output'] = np.where(new_data['output'] == 0,"Rejected","Accepted")
        final_data['Output']=new_data['output']
        return final_data

# =====================================================================================================
class LR_predict:
    def __init__(self):
        pass

    def predictor(self, file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.LR_preprocess()

        set0 = instance2.initialize_columns(data)
        set1 = instance2.drop_col(set0)
        set2 = instance2.feature_engg(set1)
        set3 = instance2.outlier_removal(set2)
        set4 = instance2.imputer(set3)
        #set4 = instance2.drop_col(set3)

        #final_data = data['RowID']
        #final_data = pd.DataFrame()
        # ============  pandas-prof. report ============================
        set4.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
        # ==============================================================
        lr_model = pickle.load(open('pickle_files/loan_risk.pkl','rb'))

        result = lr_model.predict(set4)

        set4['output'] = result
        set4['output'] = np.where(set4['output'] == 0,"Risky","Safe")
        #final_data['Output']=set4['output']
        final_data = {'RowID':[i for i in set0['RowID']],'Output':[i for i in set4['output']]}
        return pd.DataFrame(final_data)
# ===============================================================================================
class LE_predict:
    def __init__(self):
        pass

    def predictor(self, file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.LE_preprocess()
        data_final = instance2.initialize_columns(data)

        # ============  pandas-prof. report ============================
        new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
        # ==============================================================
        le_model = pickle.load(open('pickle_files/LE-DecTreeModel.pkl', 'rb'))

        result = le_model.predict(data_final)

        data_final['output'] = result
        data_final['output'] = np.where(data_final['output'] == 0, "Not Eligible", "Eligible")

        data_final['RowID'] = pd.Series([i for i in range(len(data_final['output']))])

        final_data = {'RowID': [i for i in data_final['RowID']], 'Output': [i for i in data_final['output']]}
        return pd.DataFrame(final_data)

#=======================================================================================================================
class MA_predict():
    def __init__(self):
        pass

    def predictor(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)

        instance2 = preprocessingfile.MA_preprocess()

        new_data = instance2.initialize_columns(data)
        # ============  pandas-prof. report ============================
        new_data.to_csv(r'graph_input_files\graph_data.csv', index_label=False)
        # ==============================================================
        final_data = new_data[['CONCAT']]
        new_data=instance2.drop_columns(new_data)
        model_MA=pickle.load(open('pickle_files/Mortgage_RE.pkl', 'rb'))

        result = model_MA.predict(new_data)

        final_data['output'] = result
        return final_data