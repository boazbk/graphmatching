import numpy as np

np.random.seed(10)

# null distribution

def genNull(n,p,gamma):

    A1 = np.random.rand(n,n)
    A2 = np.random.rand(n,n)
    G1 = A1 <= p*gamma
    G1 = G1 + np.zeros((n,n))
    G2 = A2 <= p*gamma
    G2 = G2 + np.zeros((n,n))

    # symmetrization

    for i in range(n):
        G1[i,i] = 0
        G2[i,i] = 0
        for j in range(n):
            G1[i,j] = G1[j,i]
            G2[i,j] = G2[j,i]
    
    return G1, G2
#    return G1, randPerm(G2,n)

# structured distribution

def genStruct(n,p,gamma):

    A = np.random.rand(n,n)
    A1 = np.random.rand(n,n)
    A2 = np.random.rand(n,n)

    B = A <= p
    B = B + np.zeros((n,n))
    G1 = A1 <= gamma
    G1 = G1 + np.zeros((n,n))
    G2 = A2 <= gamma
    G2 = G2 + np.zeros((n,n))

    G1 = np.multiply(B,G1)
    G2 = np.multiply(B,G2)


    # symmetrization

    for i in range(n):
        for j in range(n):
            G1[i,j] = G1[j,i]
            G2[i,j] = G2[j,i]
    
    return G1, G2
#    return G1, randPerm(G2,n)

# randomly permute matrix G
def randPerm(G,n):

    pi = np.random.permutation(n)
    P = np.zeros((n,n))
    Pt = np.zeros((n,n))
    for i in range(n):
        P[i,pi[i]] = 1
        Pt[pi[i],i] = 1

    Gtmp = np.matmul(P,G)
    Gout = np.matmul(Gtmp,Pt)

    return Gout
