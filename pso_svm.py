# coding=gbk

import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from utils import plot
from utils import data_handle_v1
from config import args, kernel, data_path1, data_path2
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False 

def fitness_function(position, data):
      
    X_train, X_test, y_train, y_test = data
    svclassifier = SVC(kernel=kernel, gamma=position[0], C=position[1])
    svclassifier.fit(X_train, y_train)
    y_train_pred = svclassifier.predict(X_train)
    y_test_pred = svclassifier.predict(X_test)
    return confusion_matrix(y_train, y_train_pred)[0][1] + confusion_matrix(y_train, y_train_pred)[1][0], confusion_matrix(y_test, y_test_pred)[0][1] + confusion_matrix(y_test, y_test_pred)[1][0]

def pso_svm(data):

    particle_position_vector = np.array([np.array([random.random() * 1.2, random.random() * 25]) for _ in range(args.n_particles)])
    pbest_position = particle_position_vector  
    gbest_position = np.array([float('inf'), float('inf')])  
    
    pbest_fitness_value = np.array([float('inf') for _ in range(args.n_particles)])  
    gbest_fitness_value = np.array([float('inf'), float('inf')])  
    
    velocity_vector = np.array([np.array([0, 0]) for _ in range(args.n_particles)])  
    iteration = 0
    
   
    while iteration < args.n_iterations:
       
        for i in range(args.n_particles):
            fitness_cadidate = fitness_function(particle_position_vector[i], data)
            print("error of particle-", i, "is (training, test)", fitness_cadidate, " At (gamma, c): ",
                  particle_position_vector[i])
          
            if (pbest_fitness_value[i] > fitness_cadidate[1]):
                pbest_fitness_value[i] = fitness_cadidate[1]
                pbest_position[i] = particle_position_vector[i]
            
            if (gbest_fitness_value[1] > fitness_cadidate[1]):
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]
           
            elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]
        
        for i in range(args.n_particles):
       
            new_velocity = (args.W * velocity_vector[i]) + (args.c1 * random.random()) * (
                        pbest_position[i] - particle_position_vector[i]) + (args.c2 * random.random()) * (
                                       gbest_position - particle_position_vector[i])
        
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position
        args.fitness_val_list.append(fitness_cadidate[1])
        iteration = iteration + 1
    plot(args.fitness_val_list)
    print("最优值是：%.5f,%.5f" % (gbest_fitness_value[0], gbest_fitness_value[1]))
    print("最优解是：gamma=%.5f,C=%.5f" % (gbest_position[0], gbest_position[1]))
    # print("最优解是：gamma=%.5f,c=%.5f" % (gbest_fitness_value[0],gbest_fitness_value[1]))
   

def main():
    X_train, X_test, y_train, y_test = data_handle_v1(data_path1, data_path2)
    data = [X_train, X_test, y_train, y_test]
    pso_svm(data)

if __name__ == '__main__':
    main()
