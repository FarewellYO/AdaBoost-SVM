# coding=gbk

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

from Forecast_Fault import ReadData as RD
from Forecast_Fault import CalculateMetrics as CM


# # reading data
df_train = RD.read_train_data()

print("训练集样本数量：", len(df_train))
print("训练样本中断层的数量：", len(df_train.loc[df_train["label"] == -1]))
print("训练样本中非断层的数量：", len(df_train.loc[df_train["label"] == 1]))

df_test = RD.read_test_data()
print("测试集样本数量：", len(df_test))
print("测试样本中断层的数量：", len(df_test.loc[df_test["label"] == -1]))
print("测试样本中非断层的数量：", len(df_test.loc[df_test["label"] == 1]))


# # Build the base learner--Forecast_SVM 
df_train_x, df_train_y = RD.read_x_y_train(df_train)
min_max_scaler = MinMaxScaler()
df_train_x = min_max_scaler.fit_transform(df_train_x)

svm_clf = svm.SVC(gamma=0.57628, C=21.56205, kernel='rbf')

# # Build the base learner--AdaBoost
abc_clf = AdaBoostClassifier(base_estimator=svm_clf, n_estimators=50, learning_rate=0.3, algorithm='SAMME')

# # Training model
abc_clf.fit(df_train_x, df_train_y)

df_train["prediction"] = abc_clf.predict(df_train_x)
df_train.to_csv("TrainRes.csv")

fault_num = df_train.loc[df_train['prediction'] == -1]
TP = len(df_train.loc[(df_train['label'] == -1) & (df_train['prediction'] == -1)])
TN = len(df_train.loc[(df_train['label'] == 1) & (df_train['prediction'] == 1)])
FP = len(df_train.loc[(df_train['label'] == 1) & (df_train['prediction'] == -1)])
FN = len(df_train.loc[(df_train['label'] == -1) & (df_train['prediction'] == 1)])

print("训练集中预测断层的数量：", len(fault_num))
print("训练集中TP：", TP)
print("训练集中TN：", TN)
print("训练集中FP：", FP)
print("训练集中FN：", FN)


# # Testing model
df_test_x, df_test_y = RD.read_x_y_test(df_test)
min_max_scaler = MinMaxScaler()
df_test_x = min_max_scaler.fit_transform(df_test_x)

df_test['svm_scores'] = abc_clf.decision_function(df_test_x)
df_test["prediction"] = abc_clf.predict(df_test_x)
df_test.to_csv("TestRes.csv", index=False)

T_fault_num = df_test.loc[df_test['prediction'] == -1]
T_TP = len(df_test.loc[(df_test['label'] == -1) & (df_test['prediction'] == -1)])
T_TN = len(df_test.loc[(df_test['label'] == 1) & (df_test['prediction'] == 1)])
T_FP = len(df_test.loc[(df_test['label'] == 1) & (df_test['prediction'] == -1)])
T_FN = len(df_test.loc[(df_test['label'] == -1) & (df_test['prediction'] == 1)])

print("测试集中预测断层的数量：", len(T_fault_num))
print("测试集中TP：", T_TP)
print("测试集中TN：", T_TN)
print("训测试集中FP：", T_FP)
print("测试集中FN：", T_FN)
print("__________________________")



CM.evaluation_index(T_TP, T_TN, T_FP, T_FN)

y_test = df_test['label']
y_score = df_test['svm_scores']

CM.au_index(y_test, y_score)

conf_max = CM.draw_confu_max(abc_clf, df_test_x, df_test_y)


