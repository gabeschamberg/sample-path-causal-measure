import numpy as np

###### Generate sequences #####
# N - number of sequences
# T - number of samples
# K - markov order
# params - N x K+1 matrix (None => random)
# seed - random seed to use for params (None => random)
##############################
def gen_bernoulli(N,T,K,params=None,biases=None,
                seed=None,ret_r_probs=False):

    if params is not None:
        # For each of the N sequences, need NxK coeffs
        if params.shape != (N,N,K):
            raise ValueError('params.shape must be (N,N,K)')
    else:
        params = np.random.laplace(size=(N,N,K))

    if biases is not None:
        if biases.shape != (N,):
            raise ValueError('biases.shape must be N')
    else:
        biases = np.random.randn(N)

    # create matrix for sequences
    sequences = np.zeros((N,T+K))
    probs = np.zeros((N,T))

    for t in range(K,T+K):
        history = sequences[:,t-K:t]
        for n in range(N):
            total = 0
            for m in range(N):
                total += np.dot(history[m,:],params[n,m,:])
            bias = biases[n]
            exp = np.exp(-(bias + total))
            prob = exp/(1+exp)
            probs[n,t-K] = prob
            sequences[n,t] = np.random.binomial(1,prob)
    sequences = sequences[:,K:]

    if(ret_r_probs):
        r_probs = get_marg_probs(N,T,K,biases,params)
        return (sequences,probs,r_probs)

    else:
        return (sequences,probs)

# Get the restricted probs (assuming Z and Y are iid)
def get_r_probs(N,T,K,sequences,biases,params):
    py = np.exp(-biases[1])/(1+np.exp(-biases[1]))
    pz = np.exp(-biases[2])/(1+np.exp(-biases[2]))
    for t in range(T):
        samples = sequnces[:,t]
        x = samples[0]
        y = samples[1]
        z = samples[2]


