import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    out = np.reshape(x, (x.shape[0], -1)) 
    out = out.dot(w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout) 
    db = dout.sum(axis=0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(x, 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx = dout * np.where(x > 0, 1, 0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # Compute output
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + eps)
        xn = xc / std
        out = gamma * xn + beta

        cache = (mode, x, gamma, xc, std, xn, out)

        # Update running average of mean
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        # Update running average of variance
        running_var *= momentum
        running_var += (1 - momentum) * var
    elif mode == 'test':
        # Using running mean and variance to normalize
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
        cache = (mode, x, xn, gamma, beta, std)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache
        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dx = gamma / std / N * (N * dout - xn * dgamma - dbeta)
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask = (np.random.rand(*x.shape) > p) / (1-p)
        out = x * mask        
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        dx = dout * mask
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W  = x.shape
    F, C, HH, WW = w.shape
    filt = w
    # create output ndarray to be filled
    out = np.ones((N, F, 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride)) 
    # pad on all sides, based on the params
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    for hi in range(1 + (H + 2 * pad - HH) // stride):
        for wi in range(1 + (W + 2 * pad - WW) // stride):
            s = xpad[:, :, hi*stride:hi*stride + HH, wi*stride:wi*stride+WW]
            # Make s have shape [B, 1, C, HH, WW] to broadcast multiply with filter of shape [F, C, HH, WW]
            out[:, :, hi, wi] = (s[:, None] * filt).sum(axis=(2, 3, 4)) + b    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    db = dout.sum(axis=(0, 2, 3))
    x, w, b, conv_param = cache 
    dw = np.zeros_like(w)
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W  = x.shape
    F, C, HH, WW = w.shape
    filt = w

    # dout: (N, F, H, W)
    # dw: (F, C, HH, WW)
    out = np.ones((N, F, 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride))
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx = np.zeros_like(x)

    # Explicit solution (vectorized version below)
    # This is a bit easier to understand, at the cost of looping over every dimension.
    # This can alternatively be implemented as a convolution with a transposed (rotated) filter.

    # for h in range(1 + (H + 2 * pad - HH) // stride):
    #     for w in range(1 + (W + 2 * pad - WW) // stride):
    #         for n in range(N):
    #             for f in range(F):
    #                 for c in range(C):
    #                     # dout_dw is just a slice of the actual dout_dw array
    #                     dout_dw = x[n, c, h*stride:h*stride + HH, w*stride:w*stride+WW]
    #                     dw[f, c] += (dout_dw * dout[n, f, h, w])
    #                     # dout_dx is just a slice of the actual dout_dx array
    #                     dout_dx = filt[f, c]
    #                     dx[n, c, h*stride:h*stride + HH, w*stride:w*stride+WW] += (dout_dx * dout[n, f, h, w])
    # dx = dx[:, :, pad:-pad, pad:-pad]   
 
    for h in range(1 + (H + 2 * pad - HH) // stride):
        for w in range(1 + (W + 2 * pad - WW) // stride):
            # dout_dw is just a slice of the actual dout_dw array
            dout_dw = x[:, :, h*stride:h*stride + HH, w*stride:w*stride+WW]
            dw += (dout_dw[:, None] * dout[:, :, h, w][..., None, None, None]).sum(axis=0)

            dx[:, :, h*stride:h*stride + HH, w*stride:w*stride+WW] += (filt * dout[:, :, h, w][..., None, None, None]).sum(axis=1)
    dx = dx[:, :, pad:-pad, pad:-pad]    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W  = x.shape
    out = np.ones((N, C, (H - pool_height)//stride +1, (W - pool_width)//stride +1))
    for h in range((H - pool_height)//stride +1):
        for w in range((W - pool_width)//stride +1):
           segment = x[:, :, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width] 
           out[:, :, h, w] = np.max(segment, axis=(2, 3))    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    dx = np.zeros_like(x)
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W  = x.shape
    out = np.ones((N, C, (H - pool_height)//stride +1, (W - pool_width)//stride +1))

    for h in range((H - pool_height)//stride +1):
        for w in range((W - pool_width)//stride +1):
            # segment is a rectangle of the size of the maxpool filter
            segment = x[:, :, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width] 
            # Find where the value is coming from for each square segment (for each particular channel/batch). 
            for n in range(N):
                for c in range(C):
                    best = np.unravel_index(np.argmax(segment[n, c], axis=None), segment[n, c].shape)                  
                    dx[n, c, h*stride+best[0], w*stride+best[1]] += dout[n, c, h, w]   
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    out = np.transpose(x, (0, 2, 3, 1)) 
    s = out.shape
    out = np.reshape(out, (-1, out.shape[-1])) 
    out, cache = batchnorm_forward(out, gamma, beta, bn_param)
    out = np.reshape(out, s)
    out = np.transpose(out, (0, 3, 1, 2))    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    dout_p = np.transpose(dout, (0, 2, 3, 1)) 
    s = dout_p.shape
    dout_p = np.reshape(dout_p, (-1, dout_p.shape[-1])) 
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_p, cache)
    dx = np.reshape(dx, s)
    dx = np.transpose(dx, (0, 3, 1, 2))    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
