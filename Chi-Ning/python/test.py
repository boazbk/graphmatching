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
p = 0.4         # np is the average degree
gamma = 0.85     # noise
k = 20
T = 50

error_null  = np.zeros((T,k-1))
error_struct = np.zeros((T,k-1))

deg_null = np.zeros(T)
deg_struct = np.zeros(T)

sub_null = np.zeros(T)
sub_struct = np.zeros(T)

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

    sub_null[t] = subgraphCount(G1_null,G2_null,n,k)
    sub_struct[t] = subgraphCount(G1_struct,G2_struct,n,k)
#    sub_null[t] = sub_null[t]*sub_null[t]
#    sub_struct[t] = sub_struct[t]*sub_struct[t]

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


    for i in range(k-1):
        error_null[t,i] = np.linalg.norm(np.sort(v1_null[i,:])-np.sort(v2_null[i,:]))
        error_struct[t,i] = np.linalg.norm(np.sort(v1_struct[i,:])-np.sort(v2_struct[i,:]))

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


# subgraph counting
cnt_null = 0
cnt_struct = 0
sub_middle = 0
for t in range(T):
    tmp_middle = np.absolute(sub_null[t])

    tmp_null = 0.0
    tmp_struct = 0.0

    for t in range(T):
        if np.absolute(sub_null[t]) <= tmp_middle:
            tmp_null = tmp_null+1
        if np.absolute(sub_struct[t]) > tmp_middle:
            tmp_struct = tmp_struct+1
    
    if tmp_null+tmp_struct > cnt_null+cnt_struct:
        cnt_null = tmp_null
        cnt_struct = tmp_struct
        sub_middle = tmp_middle

    tmp_middle = np.absolute(sub_struct[t])

    tmp_null = 0.0
    tmp_struct = 0.0

    for t in range(T):
        if np.absolute(sub_null[t]) <= tmp_middle:
            tmp_null = tmp_null+1
        if np.absolute(sub_struct[t]) > tmp_middle:
            tmp_struct = tmp_struct+1
    
    if tmp_null+tmp_struct > cnt_null+cnt_struct:
        cnt_null = tmp_null
        cnt_struct = tmp_struct
        sub_middle = tmp_middle


print "------- Subgraph Counting -------"
print "Threshold: ",
print sub_middle
print "[Null]: ",
print cnt_null/T
print "[Structured]",
print cnt_struct/T

