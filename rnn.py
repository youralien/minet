# A modification of rnn.py by Razvan Pascanu
import numpy as np
import theano
import theano.tensor as TT
from theano.compat.python2x import OrderedDict

# number of hidden units
n = 50
# number of input units
nin = 5
# number of output units
nout = 5

# input (where first dimension is time)
u = TT.matrix()
# target (where first dimension is time)
t = TT.matrix()
# initial hidden state of the RNN
h0 = TT.vector()
# learning rate
lr = TT.scalar()
# recurrent weights as a shared variable
W = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01).astype(np.float32))
# input to hidden layer weights
W_in = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01).astype(np.float32))
# hidden to output layer weights
W_out = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01).astype(np.float32))


# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h_tm1, W, W_in, W_out):
    h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
    y_t = TT.dot(h_t, W_out)
    return h_t, y_t

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])
# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPTT
gW, gW_in, gW_out = TT.grad(error, [W, W_in, W_out])
# training function, that computes the error and updates the weights using
# SGD.

ud = OrderedDict()
ud[W] = W - lr * gW
ud[W_in] = W_in - lr * gW_in
ud[W_out] = W_out - lr * gW_out

# h0 should be np.zeros(size)
# lr should be .01 for now, although this could be different for different updates funcs like rmsprop adagrad
fn = theano.function([h0, u, t, lr],
                     error,
                     updates=ud)


# lets train / test stuff!

trX = np.linspace(-5, 5, 101)
trY = trX ** 2 + np.random.randn(*trX.shape) * 1.3 # noise for training

plt.plot(trY, 'r.')
plt.show()

teX = np.linspace(-7, 7, 101)
teY = teX ** 2 # no noise for testing

tru = trX.reshape(-1, 1)
trt = trY.reshape(-1, 1)
teu = teX.reshape(-1, 1)
tet = teX.reshape(-1, 1)


