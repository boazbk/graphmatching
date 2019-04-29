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

np.random.seed(101)

# init

n = 200         # number of vertices
p = 0.3         # np is the average degree
gamma = 0.5     # noise
k = 4
T = 50

v1_null = np.zeros([k-1,T])
v2_null = np.zeros([k-1,T])
v3_null = np.zeros([k-1,T])
v1_struct = np.zeros([k-1,T])
v2_struct = np.zeros([k-1,T])

deg_null = np.zeros(T)
deg_struct = np.zeros(T)

(mean_est, std_est) = estCycleCount(n,k,p,gamma,100)

for t in range(T):

    (G1_null, G2_null) = genNull(n,p,gamma)
    (G3_null, G4_null) = genNull(n,p,gamma)
    (G1_struct, G2_struct) = genStruct(n,p,gamma)

    v1_null[:,t] = np.sum(cycleCount(G1_null,n,k), axis=1)
    v2_null[:,t] = np.sum(cycleCount(G2_null,n,k), axis=1)
    v3_null[:,t] = np.sum(cycleCount(G3_null,n,k), axis=1)
    v1_struct[:,t] = np.sum(cycleCount(G1_struct,n,k), axis=1)
    v2_struct[:,t] = np.sum(cycleCount(G2_struct,n,k), axis=1)
    
    d1_null = degreeCount(G1_null) 
    d2_null = degreeCount(G2_null) 
    d1_struct = degreeCount(G1_struct) 
    d2_struct = degreeCount(G2_struct) 
    # degree sequence
    deg_null[t] = np.linalg.norm(np.sort(d1_null)-np.sort(d2_null))
    deg_struct[t] = np.linalg.norm(np.sort(d1_struct)-np.sort(d2_struct))

#for i in range(k-1):
#    v1_null[i,:] = v1_null[i,:]/((i+2)*(i+2)) # divide #aut
#    v2_null[i,:] = v2_null[i,:]/((i+2)*(i+2)) # divide #aut
#    v3_null[i,:] = v3_null[i,:]/((i+2)*(i+2)) # divide #aut
#    v1_struct[i,:] = v1_struct[i,:]/((i+2)*(i+2)) # divide #aut
#    v2_struct[i,:] = v2_struct[i,:]/((i+2)*(i+2)) # divide #aut


# substract mean
#v1_null = v1_null - np.matmul(np.reshape(np.average(v3_null,axis=1),(k-1,1)),np.ones([1,T]))
#v2_null = v2_null - np.matmul(np.reshape(np.average(v3_null,axis=1),(k-1,1)),np.ones([1,T]))
#v1_struct = v1_struct - np.matmul(np.reshape(np.average(v3_null,axis=1),(k-1,1)),np.ones([1,T]))
#v2_struct = v2_struct - np.matmul(np.reshape(np.average(v3_null,axis=1),(k-1,1)),np.ones([1,T]))


for i in range(k-1):
#    v1_null[i,:] = v1_null[i,:]/np.std(v3_null[i,:])
#    v2_null[i,:] = v2_null[i,:]/np.std(v3_null[i,:])
#    v1_struct[i,:] = v1_struct[i,:]/np.std(v3_null[i,:])
#    v2_struct[i,:] = v2_struct[i,:]/np.std(v3_null[i,:])
    v1_null[i,:] = (v1_null[i,:]-mean_est[i]*np.ones(T))/std_est[i]
    v2_null[i,:] = (v2_null[i,:]-mean_est[i]*np.ones(T))/std_est[i]
    v1_struct[i,:] = (v1_struct[i,:]-mean_est[i]*np.ones(T))/std_est[i]
    v2_struct[i,:] = (v2_struct[i,:]-mean_est[i]*np.ones(T))/std_est[i]

print "[null]"
print v1_null
print "[struct]"
print v1_struct

# compute P

v_null =  v1_null*v2_null
v_struct = v1_struct*v2_struct

sub_null = np.sum(v_null,axis=0)
sub_struct = np.sum(v_struct,axis=0)
deg_null = v_null[0,:]
deg_struct = v_struct[0,:]

# print sub_null
# print sub_struct

# subgraph counting
(mean_est, std_est) = estCycleCount(n,k,p,gamma,50)
thres_sub = estThresSub(n,k,p,gamma,50,mean_est,std_est)
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

cnt_null = 0.0
cnt_struct = 0.0
for t in range(T):
    if np.absolute(sub_null[t]) <= thres_sub:
        cnt_null = cnt_null+1
    if np.absolute(sub_struct[t]) > thres_sub:
        cnt_struct = cnt_struct+1


print "------- Subgraph Counting -------"
print "Threshold: ",
print thres_sub
print "[Null]: ",
print cnt_null/T
print "[Structured]",
print cnt_struct/T


# degree sequence
cnt_null = 0
cnt_struct = 0
deg_middle = 0
for t in range(T):
    tmp_middle = np.absolute(deg_null[t])

    tmp_null = 0.0
    tmp_struct = 0.0

    for t in range(T):
        if np.absolute(deg_null[t]) <= tmp_middle:
            tmp_null = tmp_null+1
        if np.absolute(deg_struct[t]) > tmp_middle:
            tmp_struct = tmp_struct+1
    
    if tmp_null+tmp_struct > cnt_null+cnt_struct:
        cnt_null = tmp_null
        cnt_struct = tmp_struct
        deg_middle = tmp_middle

    tmp_middle = np.absolute(deg_struct[t])

    tmp_null = 0.0
    tmp_struct = 0.0

    for t in range(T):
        if np.absolute(deg_null[t]) <= tmp_middle:
            tmp_null = tmp_null+1
        if np.absolute(deg_struct[t]) > tmp_middle:
            tmp_struct = tmp_struct+1
    
    if tmp_null+tmp_struct > cnt_null+cnt_struct:
        cnt_null = tmp_null
        cnt_struct = tmp_struct
        deg_middle = tmp_middle


print "------- Degree Sequence -------"
print "[Null]: ",
print cnt_null/T
print "[Structured]",
print cnt_struct/T

# print deg_null
# print deg_struct



