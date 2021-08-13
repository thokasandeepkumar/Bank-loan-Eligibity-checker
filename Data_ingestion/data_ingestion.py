import pandas as pd
import numpy as np

class data_getter:

    def data_load(self, file):
        '''
                            Method Name: data_load
                            Description: This method loads the data from the file and convert into a pandas dataframe
                            Output: Returns a Dataframes, which is our data for training
                            On Failure: Raise Exception .
        '''
        try:
            ##This is to display the numerical values as decimals instead of scientific notations like 5.951188e+08
            pd.set_option('float_format', '{:f}'.format)
            data = pd.read_csv(file,encoding = "ISO-8859-1",error_bad_lines=False)
            return data
        except Exception as e:
            raise e


