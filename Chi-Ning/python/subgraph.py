import numpy as np;
import numpy.linalg as la;

def counts(G, list_of_cycle_length, degree_bool):
    degrees = [];
    if degree_bool:
        degrees = count_degree(G);
    
    cycles = count_subgraphs(G, list_of_cycle_length);
    return np.concatenate((degrees, cycles), axis = 1);
    
# for all functions, G stands for adj matrix
def count_degree(G):
    (n, n) = np.shape(G);
    return np.sum(G, axis = 0).reshape(n, 1);

# using nonbacktracking matrix to count short cycles
# should work for 3, 4, 5
def count_subgraphs(G, li):
    nei = neighbors(G);
    edgeindex = edgemap(nei);
    NG = non_backtracking(G, nei, edgeindex);
    ci = cycleindex(nei, edgeindex);
    result = [];
    for l in li:
        temp = count_cycle(NG, l, ci);
        result.append(temp);
    return np.asarray(result).T;

def count_cycle(NG, l, ci) :
    if (l > 2):
        result = []
        cur = la.matrix_power(NG, l - 1);
        for i in range(len(ci)):
            curlist = [cur[i1][i2] for (i1, i2) in ci[i]];
            result.append(sum(curlist));
        return result;
    else:
        return [];



def non_backtracking(G, nei, edgeindex):
    m = np.sum(G);
    result = np.zeros((2*m, 2*m));
    for (i, j) in edgeindex:
        for (u, v) in nei[int(j)]:
            if v != i:
                result[edgeindex[(i, j)]][edgeindex[(u, v)]] = 1;
    return result;

def neighbors(G):
    (n,n) = G.shape;
    result = [];
    for i in range(n):
        cur_row = G[i, :];
        temp = np.nonzero(cur_row)[0];
        ones = np.ones(len(temp)).T*i;
        cur = np.array(list(zip(ones, temp))).astype(int);
        result.append(cur);
    return result;
def edgemap(nei):
    result = {};
    count = 0;
    for i in range(len(nei)):
        for [u, v] in nei[i]:
            result[(u, v)] = count;
            count +=1;
    return result;

def cycleindex(nei, edgeindex):
    result = [];
    
    for i in range(len(nei)):
        cur = [];
        l = len(nei[i]);
        # add (i, u)(v, i) (i, v)(u, i) only once
        for j in range(l):
            [a, b] = nei[i][j];
            i1 = edgeindex[(a, b)];
            for k in range(j + 1, l):
                (u, v) = nei[i][k];
                i2 = edgeindex[(v, i)];
                cur.append((i1, i2));
        result.append(cur);
    return result;


