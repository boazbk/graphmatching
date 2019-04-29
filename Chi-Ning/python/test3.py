import numpy as np
from gen import genNull
from gen import genStruct
from algorithms import degreeSeq
from algorithms import subgraphCount
from algorithms import cycleCount
from algorithms import degreeCount
from algorithms import estCycleCount
from algorithms import estThresDeg
from algorithms import estThresSub
from algorithms import calCor

np.random.seed(101)

# init

n = 100         # number of vertices
p = 0.1         # np is the average degree
gamma = 0.5     # noise
k = 4
T = 10

cor_null = np.zeros(T)
cor_struct = np.zeros(T)

cycle1_null = np.zeros([k-1,T])
cycle2_null = np.zeros([k-1,T])
cycle_null = np.zeros([k-1,T])
cycle1_struct = np.zeros([k-1,T])
cycle2_struct = np.zeros([k-1,T])
cycle_struct = np.zeros([k-1,T])

(mean_est, std_est) = estCycleCount(n,k,p,gamma,100)

for t in range(T):

    (G1_null, G2_null) = genNull(n,p,gamma)
    (G1_struct, G2_struct) = genStruct(n,p,gamma)


    v1_null = cycleCount(G1_null,n,k)
    v2_null = cycleCount(G2_null,n,k)
    v1_struct = cycleCount(G1_struct,n,k)
    v2_struct = cycleCount(G2_struct,n,k)

    print "--- ", t, " ---"
    print np.sum(v1_null,axis=1)-mean_est
    print np.sum(v2_null,axis=1)-mean_est
    print np.sum(v1_struct,axis=1)-mean_est
    print np.sum(v2_struct,axis=1)-mean_est

    cycle1_null[:,t] = (np.sum(v1_null,axis=1)-mean_est)/std_est
    cycle2_null[:,t] = (np.sum(v2_null,axis=1)-mean_est)/std_est
    cycle1_struct[:,t] = (np.sum(v1_struct,axis=1)-mean_est)/std_est
    cycle2_struct[:,t] = (np.sum(v2_struct,axis=1)-mean_est)/std_est
    
    cycle_null[:,t] = cycle1_null[:,t]*cycle2_null[:,t]
    cycle_struct[:,t] = cycle1_struct[:,t]*cycle2_struct[:,t]

    cor_null[t] = calCor(G1_null,G2_null,n,k,mean_est,std_est)
    cor_struct[t] = calCor(G1_struct,G2_struct,n,k,mean_est,std_est)

print "[null]"
print cycle_null
print cor_null
print "[struct]"
print cycle_struct
print cor_struct
print "[mean, std]"
print mean_est
print std_est
