from __future__ import division 
import numpy as np
# Generate the data according to the specification in the homework description # for part (b)
A = np.array([[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]])  #A
phi = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])  #B
pi0 = np.array([0.5, 0.3, 0.2]) #PI
X = []
for _ in xrange(5000):     
	z = [np.random.choice([0,1,2], p=pi0)]     
	for _ in range(3):         
		z.append(np.random.choice([0,1,2], p=A[z[-1]]))     
	x = [np.random.choice([0,1], p=phi[zi]) for zi in z]   
	X.append(x)
N = [500, 1000, 2000, 5000]


# TODO: Implement Baum-Welch for estimating the parameters of the HMM

# Some utitlities for tracing our implementation below

# utilties for printing out parameters of HMM

import pandas as pd



def print_B(B):
    print(pd.DataFrame(B, columns=pos_labels, index=word_labels))
    
def print_A(A):
    print(pd.DataFrame(A, columns=pos_labels, index=pos_labels))
        


def left_pad(i, s):
    return "\n".join(["{}{}".format(' '*i, l) for l in s.split("\n")])

def pad_print(i, s):
    print(left_pad(i, s))
    
def pad_print_args(i, **kwargs):
    pad_print(i, "\n".join(["{}:\n{}".format(k, kwargs[k]) for k in sorted(kwargs.keys())])) 


def backward(params, observations):
    pi, A, B = params
    N = len(observations)
    S = pi.shape[0]
    
    beta = np.zeros((N, S))
    
    # base case
    beta[N-1, :] = 1
    
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * B[observations[i+1], s2]
    
    return (beta, np.sum(pi * B[observations[0], :] * beta[0,:]))


def forward(params, observations):
    pi, A, B = params
    N = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((N, S))
    
    # base case
    alpha[0, :] = pi * B[observations[0], :]
    
    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * B[observations[i], s2]    
    
    return (alpha, np.sum(alpha[N-1,:]))

def print_forward(params, observations):
    alpha, za = forward(params, observations)
    print(pd.DataFrame(
            alpha, 
            columns=pos_labels, 
            index=[word_labels[i] for i in observations]))

print_forward((pi, A, B), [THE, DOG, WALKED, IN, THE, PARK, END])
print_forward((pi, A, B), [THE, CAT, RAN, IN, THE, PARK, END])    



def baum_welch(training, pi, A, B, iterations, trace=False):
    pi, A, B = np.copy(pi), np.copy(A), np.copy(B)  # take copies, as we modify them
    S = pi.shape[0]

    # iterations of EM
    for it in range(iterations):
        if trace:
            pad_print(0, "for it={} in range(iterations)".format(it))
            pad_print_args(2, A=A, B=B, pi=pi, S=S)
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        B1 = np.zeros_like(B)

        for observations in training:
            if trace:
                pad_print(2, "for observations={} in training".format(observations))

            # 
            # E-Step: compute forward-backward matrices
            # 
                
            alpha, za = forward((pi, A, B), observations)
            beta, zb = backward((pi, A, B), observations)
            if trace:
                pad_print(4, """alpha, za = forward((pi, A, B), observations)\nbeta, zb = backward((pi, A, B), observations)""")
                pad_print_args(4, alpha=alpha, beta=beta, za=za, zb=zb)

            assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree ({} vs {})".format(za, zb)

            #
            # M-step: calculating the frequency of starting state, transitions and (state, obs) pairs
            #
            
            # Update PI: 
            pi1 += alpha[0, :] * beta[0, :] / za

            if trace:
                pad_print(4, "pi1 += alpha[0, :] * beta[0, :] / za")
                pad_print_args(4, pi1=pi1)
                pad_print(4, "for i in range(0, len(observations)):")
            
            # Update B (transition) matrix
            for i in range(0, len(observations)):
                # Hint: B1 can be updated similarly to PI for each row 1 
                B1[observations[i], :] += alpha[i, :] * beta[i, :] / za
                if trace:
                    pad_print(6, "B1[observations[{i}], :] += alpha[{i}, :] * beta[{i}, :] / za".format(i=i))
            if trace:
                pad_print_args(4, B1=B1)
                pad_print(4, "for i in range(1, len(observations)):")
                
            # Update A (emission) matrix
            for i in range(1, len(observations)):
                if trace: 
                    pad_print(6, "for s1 in range(S={})".format(S))
                for s1 in range(S):
                    if trace: pad_print(8, "for s2 in range(S={})".format(S))
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i - 1, s1] * A[s1, s2] * B[observations[i], s2] * beta[i, s2] / za
                        if trace: pad_print(10, "A1[{s1}, {s2}] += alpha[{i_1}, {s1}] * A[{s1}, {s2}] * B[observations[{i}], {s2}] * beta[{i}, {s2}] / za".format(s1=s1, s2=s2, i=i, i_1=i-1))
            if trace: pad_print_args(4, A1=A1)

        # normalise pi1, A1, B1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            B[s, :] = B1[s, :] / np.sum(B1[s, :])
        
    return pi, A, B

pi2, A2, B2 = baum_welch(['0','1','1','0'], pi, A, B, 10, trace=False)