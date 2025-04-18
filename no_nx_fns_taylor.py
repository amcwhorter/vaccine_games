import numpy as np
import networkx as nx
from numba import njit

@njit
def boltzmann(q, T):
    q1, q2 = q[0], q[1]
    a = np.e ** (q1 / T)
    b = np.e ** (q2 / T)
    if a + b == 0:
        x = (q2 - q1) / T

        taylor = 1/2 - 1/4 * x + 1/48 * x ** 3 - 1/480 * x ** 5    
        return taylor

    return a / (a + b)

@njit
def vec_state(i, a_t, A=np.array([[]])):
    neighbors = np.where(A[i] == 1)[0]
    vax_count = np.sum(a_t[neighbors])
    return vax_count

@njit
def vec_choose_act(i, s_t, T, Q=np.array([[[]]])):
    q = Q[i][s_t[i]]
    b = boltzmann(q, T)
    p = np.random.random()
    if p < b:
        return 0  # do not vaccinate
    else:
        return 1  # vaccinate

@njit
def epidemic(a_t, r, N, beta, gamma, A=np.array([[]])):

    S = np.where(a_t == 0)[0]
    if len(S) == 0:
        return np.ones(N) * -r
    I = np.random.choice(S, len(S)//10)
    S = np.array(list(set(S) - set(I)))
    R = np.where(a_t == 1)[0]

    payoffs = np.zeros(N)
    for i in range(N):              #could be optimized
        if i in R:
            payoffs[i] = -r
        elif i in I:
            payoffs[i] = -1

    I_sizes = []
    while I.shape != (0,):
        I_size = len(I)
        I_sizes.append(I_size)

        #contagion
        for i in I:
            neighbors = np.where(A[i] == 1)[0]
            S_neighbors = np.intersect1d(neighbors, S)
            p = np.random.random(len(S_neighbors))
            I_neighbors = S_neighbors[p < beta]
            I_new = np.concatenate((I, I_neighbors))
            S_new = np.array(list(set(S) - set(I_new)))
            for j in I_neighbors:
                payoffs[j] = -1

        #recovery
        p = np.random.random(I_size)
        R_just_recovered = I[p < gamma]
        R_new = np.concatenate((R, R_just_recovered))
        I_new = np.array(list(set(I_new) - set(R_new)))

        S = S_new
        I = I_new
        R = R_new
    return payoffs

@njit
def q_update(s_t, a_t, r_t1, s_t1, discount, alpha, N, Q=np.array([[[]]]),):
    for i in range(N):
        q_new = (1 - alpha) * Q[i, s_t[i], a_t[i]] + alpha * (r_t1[i] + discount * max(Q[i, s_t1[i], :]))
        Q[i, s_t[i], a_t[i]] = q_new
    return Q


#instantiation of Q and g
def instantiate_g(k, N):
    g = nx.random_regular_graph(k, N)
    return nx.to_numpy_array(g)

@njit
def instantiate_Q(k, N):
    return np.random.random((N, k + 1, 2))


@njit
def run_model(T, A=np.array([[]]), Q=np.array([[[]]]), discount=0.95, alpha=1, beta=0.4, gamma=0.5, r=0.2, N=100, k=4, n=10000):
    pandemic_size = np.zeros(n)
    n_vaxed = np.zeros(n)
    avg_payoff = np.zeros(n)

    a_t = np.zeros(N, dtype=np.int_)
    p = np.random.random(N)
    a_t[p > 0.5] = 1
    s_t = np.array([vec_state(i, a_t, A) for i in range(N)], dtype=np.int_)

    for t in range(n):
        r_t = epidemic(a_t, r, N, beta, gamma, A)

        #record pandemic outcomes
        pandemic_size[t] = len(np.where(r_t == -1)[0])
        avg_payoff[t] = sum(r_t)/N
        n_vaxed[t] = sum(a_t)

        #choose next action and record next state
        a_t1 = np.array([vec_choose_act(i, s_t, T, Q) for i in range(N)], dtype=np.int_)
        s_t1 = np.array([vec_state(i, a_t1, A) for i in range(N)], dtype=np.int_)

        #update q-table
        Q = q_update(s_t, a_t, r_t, s_t1, discount, alpha, N, Q)

        #increment t
        s_t = s_t1
        a_t = a_t1

    return n_vaxed, pandemic_size