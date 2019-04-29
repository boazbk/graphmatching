import numpy as np
import matplotlib.pyplot as plt
from gen import genNull
from gen import genStruct
from algorithms import degreeSeq
from algorithms import subgraphCount
from algorithms import estCycleCount
from algorithms import estPathCount
from algorithms import estThresDeg
from algorithms import estThresSub
from algorithms import estThresPath
from algorithms import calCor
from algorithms import calCorPath
from algorithms import cycleCount
from plot import plotResult

np.random.seed(2938)

# init

n = 150                             # number of vertices
p = 0.1                             # np is the average degree
gamma = np.arange(0.5, 1, 0.02)      # noise
k = 3
k1 = 4
k_path = 3
T = 100                              # number of samples

isPrint = 0

prob_deg_null = np.zeros(gamma.size)
prob_deg_struct = np.zeros(gamma.size)
prob_sub_null = np.zeros(gamma.size)
prob_sub_struct = np.zeros(gamma.size)
prob_sub1_null = np.zeros(gamma.size)
prob_sub1_struct = np.zeros(gamma.size)
prob_path_null = np.zeros(gamma.size)
prob_path_struct = np.zeros(gamma.size)

for g in range(gamma.size):
   
    # preproccessing    
    thres_deg = estThresDeg(n,k,p,gamma[g],200)
    (mean_est, std_est) = estCycleCount(n,k,p,gamma[g],200)
    thres_sub = estThresSub(n,k,p,gamma[g],200,mean_est,std_est)
    (mean_est1, std_est1) = estCycleCount(n,k1,p,gamma[g],200)
    thres_sub1 = estThresSub(n,k1,p,gamma[g],200,mean_est1,std_est1)
    (mean_est_path, std_est_path) = estPathCount(n,k_path,p,gamma[g],200)
    thres_path = estThresPath(n,k_path,p,gamma[g],200,mean_est_path,std_est_path)

    print mean_est_path, std_est_path

    deg_null = np.zeros(T)
    deg_struct = np.zeros(T)
    cor_null = np.zeros(T)
    cor_struct = np.zeros(T)
    cor1_null = np.zeros(T)
    cor1_struct = np.zeros(T)
    cor_path_null = np.zeros(T)
    cor_path_struct = np.zeros(T)

    for t in range(T):
        
        (G1_null, G2_null) = genNull(n,p,gamma[g])
        (G1_struct, G2_struct) = genStruct(n,p,gamma[g])

        deg_null[t] = degreeSeq(G1_null,G2_null)
        deg_struct[t] = degreeSeq(G1_struct,G2_struct)

        if np.absolute(deg_null[t]) > thres_deg:
            prob_deg_null[g] = prob_deg_null[g]+1.0/T
        if np.absolute(deg_struct[t]) <= thres_deg:
            prob_deg_struct[g] = prob_deg_struct[g]+1.0/T

        cor_null[t] = calCor(G1_null,G2_null,n,k,mean_est,std_est)
        cor_struct[t] = calCor(G1_struct,G2_struct,n,k,mean_est,std_est)

        if np.absolute(cor_null[t]) <= thres_sub:
            prob_sub_null[g] = prob_sub_null[g]+1.0/T
        if np.absolute(cor_struct[t]) > thres_sub:
            prob_sub_struct[g] = prob_sub_struct[g]+1.0/T

        cor1_null[t] = calCor(G1_null,G2_null,n,k1,mean_est1,std_est1)
        cor1_struct[t] = calCor(G1_struct,G2_struct,n,k1,mean_est1,std_est1)

        if np.absolute(cor1_null[t]) <= thres_sub1:
            prob_sub1_null[g] = prob_sub1_null[g]+1.0/T
        if np.absolute(cor1_struct[t]) > thres_sub1:
            prob_sub1_struct[g] = prob_sub1_struct[g]+1.0/T

        cor_path_null[t] = calCorPath(G1_null,G2_null,n,k_path,mean_est_path,std_est_path)
        cor_path_struct[t] = calCorPath(G1_struct,G2_struct,n,k_path,mean_est_path,std_est_path)

        if np.absolute(cor_path_null[t]) <= thres_path:
            prob_path_null[g] = prob_path_null[g]+1.0/T
        if np.absolute(cor_path_struct[t]) > thres_path:
            prob_path_struct[g] = prob_path_struct[g]+1.0/T


    if isPrint == 1:
        print "--- noise: ", gamma[g], " ---"
        print "[Degree sequence] ", thres_deg
        print deg_null
        print deg_struct
        print "[Subgraph Counting 3-cycle] ", thres_sub
        print cor_null
        print cor_struct
        print "[Subgraph Counting 3,4-cycle] ", thres_sub1
        print cor1_null
        print cor1_struct

# plt.subplot(221)
# plt.plot(gamma,prob_deg_null,'b',gamma,prob_sub_null,'r',gamma,prob_sub1_null,'g')

# plt.subplot(222)
# plt.plot(gamma,prob_deg_struct,'b',gamma,prob_sub_struct,'r',gamma,prob_sub1_struct,'g')

# plt.subplot(223)
plt.plot(gamma,(prob_deg_null+prob_deg_struct)/2,'b',gamma,(prob_sub_null+prob_sub_struct)/2,'r',gamma,(prob_sub1_null+prob_sub1_struct)/2,'g',gamma,(prob_path_null+prob_path_struct)/2,'y')
plt.legend(["Degree Sequence","Correlation: edge, 3-cycle","Correlation: edge, 3,4-cycle","Path"],loc=3)
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
plt.show()
