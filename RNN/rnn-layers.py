# Vanilla RNN Layers helper module
import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
	next_h, cache = None, None

    intermediate_h = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(intermediate_h)
    cache = (x, Wx, prev_h, Wh, intermediate_h)

    return next_h, cache



def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    x, Wx, prev_h, Wh, intermediate_h = cache
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    d_tanh = (1 - np.tanh(intermediate_h) **2) * dnext_h
    
    db = np.sum(d_tanh, axis=0)   
    dprev_h = np.dot(d_tanh, Wh.T)
    dWh = np.dot( prev_h.T, d_tanh)
    
    dx = np.dot(d_tanh, Wx.T)
    dWx = np.dot(x.T, d_tanh)
	
	return dx, dprev_h, dWx, dWh, db




def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    N, H = h0.shape
    h, cache = np.zeros((T, N, H)), []

    x_trans = x.transpose(1,0,2)
    
    
    hidden = h0
    for t in range(T):
        h_step, h_cache = rnn_step_forward(x_trans[t], hidden, Wx, Wh, b)
        hidden = h_step
        h[t] = h_step
        cache.append(h_cache)
    
    h = h.transpose(1,0,2)
        

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx, dh0, dWx, dWh, db = np.zeros((T,N,D)), None, None, None, None

    
    dh = dh.transpose(1, 0, 2)
    pdh = np.zeros((N,H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H,H))
    db = np.zeros((H))
    
    for t in reversed(range(T)):
        this_dh = dh[t] + pdh
        pdx, pdh, pdWx, pdWh, pdb = rnn_step_backward(this_dh, cache[t])
        
        dx[t] += pdx
        dh0 = pdh
        dWx += pdWx
        dWh += pdWh
        db += pdb
        
    return dx.transpose(1,0,2), dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    N, T = x.shape
    V, D = W.shape
    
    out, cache = None, None

    out = W[x, :]
    cache = x, W

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    N, T = x.shape
    V, D = W.shape

    dW = np.zeros((V, D))
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    N, D = x.shape
    H = prev_c.shape[1]
    
    actV = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    
    a_input = actV[:,:H]
    a_forget = actV[:, H:2*H]
    a_output = actV[:, 2*H:3*H]
    a_gate = actV[:, 3*H:]
    
    i_gate = sigmoid(a_input)
    f_gate = sigmoid(a_forget)
    o_gate = sigmoid(a_output)
    g_gate = np.tanh(a_gate)
    
    next_c = f_gate * prev_c + i_gate *g_gate
    next_h =  o_gate * np.tanh(next_c)
    
    cache = (i_gate, f_gate, o_gate, g_gate, actV, a_input, a_forget, a_output, a_gate, Wx, Wh, b, prev_h, prev_c, x, next_h, next_c)               

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

    i_gate, f_gate, o_gate, g_gate, actV, a_input, a_forget, a_output, a_gate, Wx, Wh, b, prev_h, prev_c, x, next_h, next_c = cache
    
    
    do_gate = np.tanh(next_c) * dnext_h 
    dnext_c +=  o_gate *  (1 - np.tanh(next_c) **2) *dnext_h 
    
    
    df_gate = prev_c * dnext_c
    dprev_c = f_gate * dnext_c
    di_gate = g_gate * dnext_c
    dg_gate = i_gate * dnext_c
    
    da_gate = (1 - np.tanh(a_gate)**2)  * dg_gate
    da_input = sigmoid(a_input)* (1-sigmoid(a_input)) * di_gate
    da_forget = sigmoid(a_forget)* (1-sigmoid(a_forget)) * df_gate
    da_output = sigmoid(a_output)* (1-sigmoid(a_output)) *do_gate

    
    da = np.hstack((da_input, da_forget, da_output, da_gate))
    
    dWx = np.dot(x.T, da) 
    dx = np.dot(da, Wx.T)
    dprev_h = np.dot( da,Wh.T)
    dWh = np.dot(prev_h.T , da)
    db = np.sum(da, axis=0)
    

    return dx, dprev_h, dprev_c, dWx, dWh, db
