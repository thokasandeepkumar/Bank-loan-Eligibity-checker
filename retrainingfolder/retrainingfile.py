from preprocessingfolder  import preprocessingfile
import pandas as pd
import numpy as np
import joblib
from Data_ingestion import data_ingestion
class retrain:
    def __init__(self):
        pass

    def retrainer(self,file):
        instance1 = data_ingestion.data_getter()
        data = instance1.data_load(file)
        instance2 = preprocessingfile.retrain()
        set0 = instance2.initialize_columns(data)
        set1 = instance2.drop_columns(set0)
        new_data = instance2.obj_to_cat(set2)
        x,y = instance2.smote(new_data)
        instance2.model(x,y)
