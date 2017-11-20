import numpy as np

###### Generate sequences #####
# N - number of sequences
# T - number of samples
# K - markov order
# params - N x K+1 matrix (None => random)
# seed - random seed to use for params (None => random)
##############################
def gen_bernoulli(N,T,K,params=None,biases=None,seed=None):

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

    return (sequences,probs)