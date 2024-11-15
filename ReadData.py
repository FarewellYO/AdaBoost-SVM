# coding=gbk

import tkinter as tk
import pandas as pd
import numpy as np


from tkinter import filedialog
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def read_train_data():
    
    train_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    try:
        
        df_train = pd.read_csv(train_file_path, index_col=0)
        
        print("Train data loading success!")
        
    except Exception as e:
        
        print("Train data Error:", e)
    
    return df_train

def read_test_data():
    
    test_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    try:
        
        df_test = pd.read_csv(test_file_path)
        
        print("Test data loading success!")
        
    except Exception as e:
        
        print("Test data Error:", e)
    
    return df_test

def read_forecast():
    
    forecast_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    try:
        
        df_forecast = pd.read_csv(forecast_file_path, index_col=0)
        
        print("Forecast data loading success!")
        
    except Exception as e:
        
        print("Forecast data Error:", e)
        
    return df_forecast

def read_x_y_train(df_train):
    
    try:
        
        TraF1 = pd.DataFrame(df_train, columns=["dip"])
        TraF2 = pd.DataFrame(df_train, columns=["var"])
        TraF3 = pd.DataFrame(df_train, columns=["Ins_Fre"])
        TraF4 = pd.DataFrame(df_train, columns=["Ins_phase"])
        TraF6 = pd.DataFrame(df_train, columns=["Max_Amp"])
        
        df_train_x = np.column_stack((TraF1, TraF2, TraF3, TraF4, TraF5))  
        min_max_scaler = MinMaxScaler()
        
        df_train_x = min_max_scaler.fit_transform(df_train_x)
        df_train_y = df_train['label']
        
        return df_train_x, df_train_y
    
    except Exception as e:
        
        print("Feature and label data Error:", e)
        
def read_x_y_test(df_test):
    
    try:
        
        TesF1 = pd.DataFrame(df_test, columns=["dip"])
        TesF2 = pd.DataFrame(df_test, columns=["var"])
        TesF3 = pd.DataFrame(df_test, columns=["Ins_Fre"])
        TesF4 = pd.DataFrame(df_test, columns=["Ins_phase"])
        TesF5 = pd.DataFrame(df_test, columns=["Max_Amp"])

        df_test_x = np.column_stack((TesF1, TesF2, TesF3, TesF4, TesF5)) 
        min_max_scaler = MinMaxScaler()

        df_test_x = min_max_scaler.fit_transform(df_test_x)
        df_test_y = df_test['label']
        
        return df_test_x, df_test_y
    
    except Exception as e:
        
        print("Feature and label data Error:", e)
        
def read_forecast_x(df_forecast):
    
    try:
        
        ForF1 = pd.DataFrame(df_forecast, columns=["dip"])
        ForF2 = pd.DataFrame(df_forecast, columns=["var"])
        ForF3 = pd.DataFrame(df_forecast, columns=["Ins_Fre"])
        ForF4 = pd.DataFrame(df_forecast, columns=["Ins_phase"])
        ForF5 = pd.DataFrame(df_forecast, columns=["Max_Amp"])
        
        df_feature_x = np.column_stack((ForF1, ForF2, ForF3, ForF4, ForF5)) 
        
        min_max_scaler = MinMaxScaler()
        
        df_feature_x = min_max_scaler.fit_transform(df_feature_x)
        
        return df_feature_x
    
    except Exception as e:
        
        print("Forecast feature  data Error:", e)
        
        return None

root = tk.Tk()
root.withdraw()