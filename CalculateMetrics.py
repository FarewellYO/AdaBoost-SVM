# coding=gbk

import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve


def draw_confu_max(Clf, Test_x, Test_y):
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制混淆矩阵  
    disp = plot_confusion_matrix(Clf, Test_x, Test_y,
                             display_labels=['断层 -1', '非断层 1'],
                             cmap=plt.cm.Blues,
                             normalize=None)
    
    # 显示图标
    plt.title('Confusion Matrix')  
    plt.show()

def evaluation_index(TP, TN, FP, FN):
    
    Accu = (TP + TN) / (TP + TN + FP + FN)
    
    Precision=TP/ (TP +FP)
    
    Recall=TP / (TP + FN) #sensitivity
    
    Specificity = TN / (TN + FP) 
    
    Err=1-Accu
    
        
    print('准确率 = ',Accu)
    print('错误率=',Err)
    print('精确率 = ',Precision)
    print('敏感度 = ', Recall)
    print('特异性 = ',Specificity)
    
    
    
def au_index(Test_y,Score_y):
    
    
    roc_auc = roc_auc_score(Test_y, Score_y)
    print("roc曲线下面积:", roc_auc)
    
  
    
    
