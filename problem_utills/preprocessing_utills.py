from application_logging import logger
import numpy as np
import time
import os
class utills:

    def __init__(self):
        self.logger = logger.App_Logger()
    def convert_current_year_job_to_cat(self,data):
        Years = []
        for i in data['Years in current job']:

            if i == '8 years':
                Years.append(8)
            elif i == '6 years':
                Years.append(6)
            elif i == '3 years':
                Years.append(3)
            elif i == '5 years':
                Years.append(5)
            elif i == '< 1 year':
                Years.append(0.8)
            elif i == '2 years':
                Years.append(2)
            elif i == '4 years':
                Years.append(4)
            elif i == '9 years':
                Years.append(9)
            elif i == '7 years':
                Years.append(7)
            elif i == '1 year':
                Years.append(1)
            else:

                Years.append(10)
        return Years

    def handle_incorrect_credit_score(self,data):
        Cr_Scr = []
        file = open("Preprocessing_log/preprocessing_log.txt", 'a+')
        message = "handling the invalid credit score started"
        self.logger.log(file, message)
        for i in data['Credit Score']:
            if np.isnan(i):
                Cr_Scr.append(i)
            else:
                if float(str(i)[:3]) > 900:
                    Cr_Scr.append(900.0)
                else:
                    Cr_Scr.append(float(str(i)[:3]))
        # self.logger.log(file,"invalid credit score has been handled successfully")
        # file.close()
        return Cr_Scr

    def saveResult_result(self,Result_path="Preprocessed_Data"):
        os.makedirs(Result_path, exist_ok=True)
        fileName = time.strftime("final_Result_%Y_%m_%d_%H_%M_%S_.csv")
        UpdatedDataSet_path = os.path.join(Result_path, fileName)
        print(f"your UpdatedDataSet will be saved at the following location\n{UpdatedDataSet_path}")
        return UpdatedDataSet_path

