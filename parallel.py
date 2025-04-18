import no_nx_fns_taylor as fn
import numpy as np
import pickle as pkl
import dask 


if __name__ == '__main__':
    from dask.distributed import Client, progress
    
    client = Client(threads_per_worker=2, n_workers=8)
    print(client.dashboard_link)

    # parameter values
    alpha = .1    #learning rate
    beta = 0.4      #transmission rate
    gamma = 0.1     #recovery rate
    discount = 0.8  #discount factor
    r = 0.1         #payoff ratio
    T = 0.05        #Starting Temp
    N = 100         #population size
    k = 4           #degree
    n = 100      #iterations

    A = fn.instantiate_g(k, N)
    Q1 = fn.instantiate_Q(k, N)
    Q2 = fn.instantiate_Q(k, N)

    
    
    rs = np.linspace(-1, 1, 201)


    def forward(T, A, Q, discount, alpha, beta, gamma, r, N, k, n, i):
        vaxed = []
        pands = []
        for r in rs:
            n_vaxed, pandemic_size = fn.run_model(T, A, Q, discount, alpha, beta, gamma, r, N, k, n)
            vaxed.append(np.mean(n_vaxed[6000:]))
            pands.append(np.mean(pandemic_size[6000:]))
        
        return vaxed, pands
    

    def backward(T, A, Q, discount, alpha, beta, gamma, r, N, k, n, i):

        rs_flipped = np.flip(rs)
        vaxed_flipped = []
        pands_flipped = []

        for r in rs_flipped:
            n_vaxed, pandemic_size = fn.run_model(T, A, Q, discount, alpha, beta, gamma, r, N, k, n)
            vaxed_flipped.append(np.mean(n_vaxed[6000:]))
            pands_flipped.append(np.mean(pandemic_size[6000:]))

        return vaxed_flipped, pands_flipped



    lazy_results = [dask.delayed(forward)(T, A, Q1, discount, alpha, beta, gamma, r, N, k, n, i) for i in range(100)] 
    results = dask.compute(lazy_results)


#    with open('pickles_taylor/fig4/neg/T=0.05/1.pkl', 'wb') as f:
#        pkl.dump(results, f)