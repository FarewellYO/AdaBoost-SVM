# coding=gbk

data_path1 = 'Train.csv'
data_path2 = 'Test.csv'

# ����Ⱥ�㷨��������
class args:
    W = 0.5  # ����Ȩ�� 0.5
    c1 = 0.2  # �ֲ�ѧϰ���� 0.2 
    c2 = 0.5  # ȫ��ѧϰ���� 0.5
    n_iterations = 50 # ��������
    n_particles = 20  # ������
    fitness_val_list = []

# SVM����
kernel = 'rbf'  # ["linear","poly","rbf","sigmoid"]
