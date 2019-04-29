import numpy as np
import matplotlib.pyplot as plt
from gen import genNull
from gen import genStruct
from algorithms import degreeCount
from algorithms import cycleCount
from algorithms import pathCount
from algorithms import estCycleCount
from algorithms import estPathCount
from algorithms import estThresDeg
from algorithms import estThresSub
from algorithms import estThresPath
from algorithms import calCor
from algorithms import calCorPath
from algorithms import cycleCount
from algorithms import distMatrix
from plot import plotResult
from scipy.optimize import linear_sum_assignment

np.random.seed(2938)

# init

n = 150                             # number of vertices
p = 0.1                             # np is the average degree
gamma = np.arange(0.8, 1, 0.02)      # noise
k = 3
k1 = 4
k_path = 3
T = 50                              # number of samples

isPrint = 0

prob_deg_struct = np.zeros(gamma.size)
prob_sub_struct = np.zeros(gamma.size)
prob_sub1_struct = np.zeros(gamma.size)
prob_path_struct = np.zeros(gamma.size)

for g in range(gamma.size):
   
    # preproccessing    
    (mean_est, std_est) = estCycleCount(n,k,p,gamma[g],100)
    (mean_est1, std_est1) = estCycleCount(n,k1,p,gamma[g],100)
    (mean_est_path, std_est_path) = estPathCount(n,k_path,p,gamma[g],100)

    deg_struct = np.zeros(T)
    cor_struct = np.zeros(T)
    cor1_struct = np.zeros(T)
    path_struct = np.zeros(T)

    for t in range(T):
        
        (G1_struct, G2_struct) = genStruct(n,p,gamma[g])

        # degree sequence
        d1_struct = degreeCount(G1_struct)
        d2_struct = degreeCount(G2_struct)

        C_deg_struct = distMatrix(d1_struct,d2_struct,n,1)

        (row_ind, col_ind) = linear_sum_assignment(C_deg_struct)
        for i in range(n):
            if row_ind[i] == col_ind[i]:
                deg_struct[t] = deg_struct[t]+1.0

        # degree + 3-cycle
        d1_struct = cycleCount(G1_struct,n,k)
        d2_struct = cycleCount(G2_struct,n,k)

        for i in range(k-1):
            d1_struct[i,:] = (d1_struct[i,:]-mean_est[i]*np.ones(n))/std_est[i]
            d2_struct[i,:] = (d2_struct[i,:]-mean_est[i]*np.ones(n))/std_est[i]

        C_deg_struct = distMatrix(d1_struct,d2_struct,n,k)

        (row_ind, col_ind) = linear_sum_assignment(C_deg_struct)
        for i in range(n):
            if row_ind[i] == col_ind[i]:
                cor_struct[t] = cor_struct[t]+1.0

        # degree +3,4-cycle
        d1_struct = cycleCount(G1_struct,n,k1)
        d2_struct = cycleCount(G2_struct,n,k1)

        for i in range(k1-1):
            d1_struct[i,:] = (d1_struct[i,:]-mean_est1[i]*np.ones(n))/std_est1[i]
            d2_struct[i,:] = (d2_struct[i,:]-mean_est1[i]*np.ones(n))/std_est1[i]

        C_deg_struct = distMatrix(d1_struct,d2_struct,n,k1)

        (row_ind, col_ind) = linear_sum_assignment(C_deg_struct)
        for i in range(n):
            if row_ind[i] == col_ind[i]:
                cor1_struct[t] = cor1_struct[t]+1.0

        # path
        d1_struct = pathCount(G1_struct,n,k_path)
        d2_struct = pathCount(G2_struct,n,k_path)

        for i in range(k_path):
            d1_struct[i,:] = (d1_struct[i,:]-mean_est_path[i]*np.ones(n))/std_est_path[i]
            d2_struct[i,:] = (d2_struct[i,:]-mean_est_path[i]*np.ones(n))/std_est_path[i]

        C_deg_struct = distMatrix(d1_struct,d2_struct,n,k_path)
        
        (row_ind, col_ind) = linear_sum_assignment(C_deg_struct)
        
        for i in range(n):
            if row_ind[i] == col_ind[i]:
                path_struct[t] = path_struct[t]+1.0

    print "[noise: ", gamma[g], "]" 
    print deg_struct
    print cor_struct
    print cor1_struct
    print path_struct

    prob_deg_struct[g] = np.mean(deg_struct)/n
    prob_sub_struct[g] = np.mean(cor_struct)/n
    prob_sub1_struct[g] = np.mean(cor1_struct)/n
    prob_path_struct[g] = np.mean(path_struct)/n

plt.plot(gamma,prob_deg_struct,'b',gamma,prob_sub_struct,'r',gamma,prob_sub1_struct,'g',gamma,prob_path_struct,'y')
plt.legend(["Degree Sequence","Correlation: edge, 3-cycle","Correlation: edge, 3,4-cycle","Path"],loc=2)
axes = plt.gca()
axes.set_xlim([0.7,1])
axes.set_ylim([0,1])
plt.show()



