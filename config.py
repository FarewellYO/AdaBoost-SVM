# coding=gbk

data_path1 = 'Train.csv'
data_path2 = 'Test.csv'

# 粒子群算法参数配置
class args:
    W = 0.5  # 惯性权重 0.5
    c1 = 0.2  # 局部学习因子 0.2 
    c2 = 0.5  # 全局学习因子 0.5
    n_iterations = 50 # 迭代次数
    n_particles = 20  # 粒子数
    fitness_val_list = []

# SVM配置
kernel = 'rbf'  # ["linear","poly","rbf","sigmoid"]
