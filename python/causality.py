from numpy import log2

def binary_causality(c_probs,r_probs):
    c = []
    for c_prob,r_prob in zip(c_probs,r_probs):
        c.append(bernoulli_kl(c_prob,r_prob))
    return c

def bernoulli_kl(p,q):
    return p*log2(p/q) + (1-p)*log2((1-p)/(1-q))