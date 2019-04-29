import numpy as np
from gen import genNull
from gen import genStruct
from subgraph import counts

# degree sequence
def degreeSeq(G1,G2):

    d1 = degreeCount(G1)
    d2 = degreeCount(G2)

    return np.linalg.norm(np.sort(d1)-np.sort(d2))

# subgraph counting: degree sequence + 3-cycle
def subgraphCount(G1,G2,n,k):

    v1 = cycleCount(G1,n,k)
    v2 = cycleCount(G2,n,k)
    
    out = 0
    # normalize each row
    for i in range(k-1):
        v1[i,:] = v1[i,:] - np.average(v1[i,:])*np.ones(n)
        v2[i,:] = v2[i,:] - np.average(v2[i,:])*np.ones(n)
        
        out = out + np.sum(v1[i,:])*np.sum(v2[i,:])

    return out

#    return np.linalg.norm(match(v1,v2,n,k))
#    return np.linalg.norm(v1-v2)

# def subgraphCountThres(G1)


def pathCount(G,n,k):

    v = np.zeros((k,n))
    A = G

    for t in range(k):
        for i in range(n):
            v[t,i] = np.count_nonzero(A[i,:])

        A = np.matmul(A,G)
        
    return v

def degreeCount(G):

    v = G.sum(axis=1)

    return v

def yukicycleCount(G,n,k):

    return counts(G,range(3,k+1),1).reshape(k-1,n)

def cycleCount(G,n,k):

    d = np.zeros((k-1,n)) # the ith row records diagonal of G^{i+2}
    c = np.zeros((k-1,n)) # the ith row records the number of (directed) (i+2)-cycle
    
    A = np.matmul(G,G)
    d[0,:] = np.diag(A)

    for i in range(k-2):
        A = np.matmul(A,G)
        d[i+1,:] = np.diag(A)
    
    c[0,:] = degreeCount(G)
    c[1,:] = d[1,:]
    
    for i in range(2,k-1):
        c[i,:] = d[i,:]
        for j in range(i-1):
            c[i,:] = c[i,:] - c[j,:]*d[i-j-2,:]

    if k >= 4:
        for i in range(n):
            c[2,i] = c[2,i] - np.inner(G[i,:],c[0,:]-np.ones(n))
#            c[2,i] = c[2,i] - np.inner(G[i,:],d[0,:]) + d[0,i]

    return c

# estimate mean and variance of cycle count
def estCycleCount(n,k,p,gamma,T):
    
    c = np.zeros([T,k-1])
    
    for t in range(T):
        (G1, G2) = genNull(n,p,gamma)
        c[t,:] = np.sum(cycleCount(G1,n,k),axis=1)
#        c[t,:] = cycleCount(G1,n,k)

    mean = np.zeros(k-1)
    std = np.zeros(k-1)

    for i in range(k-1):
        mean[i] = np.average(c[:,i])
        std[i] = np.std(c[:,i])

    return mean, std

# estimate mean and variance of path bound
def estPathCount(n,k,p,gamma,T):
    
    c = np.zeros([T,k])
    
    for t in range(T):
        (G1, G2) = genNull(n,p,gamma)
        c[t,:] = np.sum(pathCount(G1,n,k),axis=1)

    mean = np.zeros(k)
    std = np.zeros(k)

    for i in range(k):
        mean[i] = np.average(c[:,i])
        std[i] = np.std(c[:,i])

    return mean, std


# calculate correlation
def calCor(G1,G2,n,k,mean,std):

    c1 = np.sum(cycleCount(G1,n,k),axis=1)
    c2 = np.sum(cycleCount(G2,n,k),axis=1)
#    c1 = cycleCount(G1,n,k)
#    c2 = cycleCount(G2,n,k)

    out = 0
    for i in range(0,k-1):
        c1[i] = (c1[i]-mean[i])/std[i]
        c2[i] = (c2[i]-mean[i])/std[i]
        out = out + c1[i]*c2[i]

    return out

# calculate path correlation
def calCorPath(G1,G2,n,k,mean,std):

    c1 = np.sum(pathCount(G1,n,k),axis=1)
    c2 = np.sum(pathCount(G2,n,k),axis=1)

    out = 0
    for i in range(0,k):
        c1[i] = (c1[i]-mean[i])/std[i]
        c2[i] = (c2[i]-mean[i])/std[i]
        out = out + c1[i]*c2[i]

    return out

# estimate threshold for degree sequence
def estThresDeg(n,k,p,gamma,T):
    deg_null = np.zeros(T)
    deg_struct = np.zeros(T)

    for t in range(T):
        (G1_null, G2_null) = genNull(n,p,gamma)
        (G1_struct, G2_struct) = genStruct(n,p,gamma)

        deg_null[t] = degreeSeq(G1_null,G2_null)
        deg_struct[t] = degreeSeq(G1_struct,G2_struct)

    thres = estThres(deg_struct,deg_null,T)

    return thres

# estimate threshold for subgraph counting
def estThresSub(n,k,p,gamma,T,mean,std):

    cor_null = np.zeros(T)
    cor_struct = np.zeros(T)

    for t in range(T):
        (G1_null, G2_null) = genNull(n,p,gamma)
        (G1_struct, G2_struct) = genStruct(n,p,gamma)

        cor_null[t] = calCor(G1_null,G2_null,n,k,mean,std)
        cor_struct[t] = calCor(G1_struct,G2_struct,n,k,mean,std)

    thres = estThres(cor_null,cor_struct,T)

    return thres

# estimate threshold for path counting
def estThresPath(n,k,p,gamma,T,mean,std):

    cor_null = np.zeros(T)
    cor_struct = np.zeros(T)

    for t in range(T):
        (G1_null, G2_null) = genNull(n,p,gamma)
        (G1_struct, G2_struct) = genStruct(n,p,gamma)

        cor_null[t] = calCorPath(G1_null,G2_null,n,k,mean,std)
        cor_struct[t] = calCorPath(G1_struct,G2_struct,n,k,mean,std)

    thres = estThres(cor_null,cor_struct,T)

    return thres

# estimate threshold
def estThres(seq_null,seq_struct,T):

    thres = 0
    cnt_null = 0
    cnt_struct = 0
    for t in range(T):
        tmp_middle = np.absolute(seq_null[t])

        tmp_null = 0.0
        tmp_struct = 0.0

        for t in range(T):
            if np.absolute(seq_null[t]) <= tmp_middle:
                tmp_null = tmp_null+1
            if np.absolute(seq_struct[t]) > tmp_middle:
                tmp_struct = tmp_struct+1
        
        if tmp_null+tmp_struct > cnt_null+cnt_struct:
            cnt_null = tmp_null
            cnt_struct = tmp_struct
            thres = tmp_middle

        tmp_middle = np.absolute(seq_struct[t])

        tmp_null = 0.0
        tmp_struct = 0.0

        for t in range(T):
            if np.absolute(seq_null[t]) <= tmp_middle:
                tmp_null = tmp_null+1
            if np.absolute(seq_struct[t]) > tmp_middle:
                tmp_struct = tmp_struct+1
        
        if tmp_null+tmp_struct > cnt_null+cnt_struct:
                cnt_null = tmp_null
                cnt_struct = tmp_struct
                thres = tmp_middle
    
    return thres

# compute the distance matrix
def distMatrix(v1,v2,n,k):

    C = np.zeros((n,n))
    
    if k == 1:
        for i in range(n):
            for j in range(n):
                C[i,j] = np.linalg.norm(v1[i]-v2[j])
    else:
        for i in range(n):
            for j in range(n):
                C[i,j] = np.linalg.norm(v1[:,i]-v2[:,j])

    return C



