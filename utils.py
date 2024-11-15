# coding=gbk

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def plot(fitness_val_list):
    
    plt.figure(figsize=(10, 7), dpi=200)
    plt.plot(fitness_val_list, color='#1f0954', lw=2)  
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('粒子群适应度趋势')
    
    
    x_major_locator = MultipleLocator(15)
   
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-1, 52)
      
    plt.gcf()  
    plt.grid(ls='-.', linewidth=0.5) 
    plt.show()

def data_handle_v1(csv_data_path1, csv_data_path2):
    df_train = pd.read_csv(csv_data_path1, engine='python')
    
    fea1 = pd.DataFrame(df_train, columns=["dip"])
    fea2 = pd.DataFrame(df_train, columns=["var"])
    fea3 = pd.DataFrame(df_train, columns=["Ins_Fre"])
    fea4 = pd.DataFrame(df_train, columns=["Ins_phase"])
    fea5 = pd.DataFrame(df_train, columns=["Max_Amp"])
   
    df_train_x = np.column_stack((fea1, fea2, fea3, fea4, fea5))
    min_max_scaler = MinMaxScaler()
    
    X_train = min_max_scaler.fit_transform(df_train_x)
    y_train = df_train['label']
    
    df_test = pd.read_csv(csv_data_path2, engine='python')
    
    Tfea1 = pd.DataFrame(df_test, columns=["dip"])
    Tfea2 = pd.DataFrame(df_test, columns=["var"])
    Tfea3 = pd.DataFrame(df_test, columns=["Ins_Fre"])
    Tfea4 = pd.DataFrame(df_test, columns=["Ins_phase"])
    Tfea5 = pd.DataFrame(df_test, columns=["Max_Amp"])

    df_test_x = np.column_stack((Tfea1, Tfea2, Tfea3, Tfea4, Tfea5))
    X_test = min_max_scaler.transform(df_test_x)
    y_test = df_test['label']

    return X_train, X_test, y_train, y_test


