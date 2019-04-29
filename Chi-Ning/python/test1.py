import numpy as np
from gen import genNull
from gen import genStruct
from algorithms import degreeSeq
from algorithms import subgraphCount
from algorithms import cycleCount
from algorithms import degreeCount
from algorithms import match

np.random.seed(101)

# init

n = 100         # number of vertices
p = 0.6         # np is the average degree
gamma = 0.85     # noise
k = 3
T = 100

error_null  = np.zeros(T)
error_struct = np.zeros(T)

deg_null = np.zeros(T)
deg_struct = np.zeros(T)

for t in range(T):

    (G1_null, G2_null) = genNull(n,p,gamma)
    (G1_struct, G2_struct) = genStruct(n,p,gamma)

    v1_null = cycleCount(G1_null,n,k)
    v2_null = cycleCount(G2_null,n,k)
    v1_struct = cycleCount(G1_struct,n,k)
    v2_struct = cycleCount(G2_struct,n,k)

    d1_null = degreeCount(G1_null) 
    d2_null = degreeCount(G2_null) 
    d1_struct = degreeCount(G1_struct) 
    d2_struct = degreeCount(G2_struct) 

    # degree sequence
    deg_null[t] = np.linalg.norm(np.sort(d1_null)-np.sort(d2_null))
    deg_struct[t] = np.linalg.norm(np.sort(d1_struct)-np.sort(d2_struct))

    # normalize each row
    for i in range(k-1):
        v1_null[i,:] = v1_null[i,:] - np.average(v1_null[i,:])*np.ones(n)
        v2_null[i,:] = v2_null[i,:] - np.average(v2_null[i,:])*np.ones(n)
        v1_null[i,:] = v1_null[i,:]/np.std(v1_null[i,:])
        v2_null[i,:] = v2_null[i,:]/np.std(v2_null[i,:])
        v1_struct[i,:] = v1_struct[i,:] - np.average(v1_struct[i,:])*np.ones(n)
        v2_struct[i,:] = v2_struct[i,:] - np.average(v2_struct[i,:])*np.ones(n)
        v1_struct[i,:] = v1_struct[i,:]/np.std(v1_struct[i,:])
        v2_struct[i,:] = v2_struct[i,:]/np.std(v2_struct[i,:])

    pi1_null = np.argsort(d1_null)
    pi2_null = np.argsort(d2_null)
    pi1_struct = np.argsort(d1_struct)
    pi2_struct = np.argsort(d2_struct)
    
    # fixing pi
    for i in range(n):
        for j in range(max(0,i-2),min(n,i+2)):
#        for j in range(n):
            delta_ii = np.linalg.norm(v1_null[:,pi1_null[i]]-v2_null[:,pi2_null[i]])
            delta_ij = np.linalg.norm(v1_null[:,pi1_null[i]]-v2_null[:,pi2_null[j]])
            delta_ji = np.linalg.norm(v1_null[:,pi1_null[j]]-v2_null[:,pi2_null[i]])
            delta_jj = np.linalg.norm(v1_null[:,pi1_null[j]]-v2_null[:,pi2_null[j]])
            if delta_ii+delta_jj > delta_ij+delta_ji:
                tmp = pi1_null[i]
                pi1_null[i] = pi1_null[j]
                pi1_null[j] = tmp

    for i in range(n):
        for j in range(max(0,i-2),min(n,i+2)):
#        for j in range(n):
            delta_ii = np.linalg.norm(v1_struct[:,pi1_struct[i]]-v2_struct[:,pi2_struct[i]])
            delta_ij = np.linalg.norm(v1_struct[:,pi1_struct[i]]-v2_struct[:,pi2_struct[j]])
            delta_ji = np.linalg.norm(v1_struct[:,pi1_struct[j]]-v2_struct[:,pi2_struct[i]])
            delta_jj = np.linalg.norm(v1_struct[:,pi1_struct[j]]-v2_struct[:,pi2_struct[j]])
            if delta_ii+delta_jj > delta_ij+delta_ji:
                tmp = pi1_struct[i]
                pi1_struct[i] = pi1_struct[j]
                pi1_struct[j] = tmp

    # compare

    error_null[t] = np.linalg.norm(v1_null[:,pi1_null]-v2_null[:,pi2_null])
    error_struct[t] = np.linalg.norm(v1_struct[:,pi1_struct]-v2_struct[:,pi2_struct])


# degree sequence
deg_middle = (np.average(deg_null)+np.average(deg_struct))/2

cnt_null = 0.0
cnt_struct = 0.0

for t in range(T):
    if deg_null[t] > deg_middle:
        cnt_null = cnt_null+1
    if deg_struct[t] < deg_middle:
        cnt_struct = cnt_struct+1

print "------- Degree Sequence -------"
print "[Null]: ",
print cnt_null/T
print "[Structured]",
print cnt_struct/T

# print deg_null
# print deg_struct

# subgraph counting
error_middle = (np.average(error_null)+np.average(error_struct))/2

cnt_null = 0.0
cnt_struct = 0.0

for t in range(T):
    if error_null[t] < error_middle:
        cnt_null = cnt_null+1
    if error_struct[t] > error_middle:
        cnt_struct = cnt_struct+1

print "------- Subgraph Counting -------"
print "[Null]: ",
print cnt_null/T
print "[Structured]",
print cnt_struct/T

# print error_null
# print error_struct
