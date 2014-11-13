import cPickle
import gzip
import tempfile
import os

import numpy as np
import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict


def fast_dropout(rng, X):
    """ Multiply activations by N(1,1) """
    seed = rng.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=X.shape, avg=1., dtype=theano.config.floatX)
    return X * mask


def shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=np.zeros(shape, dtype=theano.config.floatX),
                         name=name, borrow=True)


def load_roland():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'roland'
    data_path = os.path.join(data_path, dataset)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        import urllib
        urls = {'train_images.txt': 'www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/train_images.txt',
                'test_images.txt': 'www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/test_images.txt',
                'train_labels.txt': 'www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/train_labels.txt',
                'test_labels.txt': 'www.iro.umontreal.ca/~memisevr/teaching/ift3395_2014/devoirs/test_labels.txt'}
        for fname, url in urls.items():
            data_file = os.path.join(data_path, fname)
            print('Downloading data from %s' % url)
            urllib.urlretrieve(url, data_file)

    print('... loading data')
    train_set_x = np.loadtxt(os.path.join(data_path, 'train_images.txt'),
                             delimiter=',').astype(theano.config.floatX)
    train_set_y = np.loadtxt(os.path.join(data_path, 'train_labels.txt'),
                             delimiter=',').argmax(axis=1).astype('int32')
    test_set_x = np.loadtxt(os.path.join(data_path, 'test_images.txt'),
                            delimiter=',').astype(theano.config.floatX)
    test_set_y = np.loadtxt(os.path.join(data_path, 'test_labels.txt'),
                            delimiter=',').argmax(axis=1).astype('int32')
    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_mnist():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'mnist.pkl.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)) and dataset == 'mnist.pkl.gz':
        import urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the name of the dataset (here MNIST)
    '''

    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        raise AttributeError("File not found at path %s" % data_file)

    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class TrainingMixin(object):
    def get_sgd_trainer(self, X_sym, y_sym, params, cost, learning_rate):
        """ Returns a simple sgd trainer."""
        gparams = T.grad(cost, params)
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - learning_rate * gparam

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn

    def get_adagrad_trainer(self, X_sym, y_sym, params, cost, learning_rate,
                            adagrad_param):
        gparams = T.grad(cost, params)
        self.accumulated_gradients_ = []
        accumulated_gradients_ = self.accumulated_gradients_

        for layer in self.layers_:
            accumulated_gradients_.extend([shared_zeros(p.shape.eval(),
                                           'accumulated_gradient')
                                           for p in layer.params])
        updates = OrderedDict()
        for agrad, param, gparam in zip(accumulated_gradients_,
                                        params, gparams):
            ag = agrad + gparam * gparam
            # TODO: Norm clipping
            updates[param] = param - (learning_rate / T.sqrt(
                ag + adagrad_param)) * gparam
            updates[agrad] = ag

        train_fn = theano.function(inputs=[X_sym, y_sym],
                                   outputs=cost,
                                   updates=updates)
        return train_fn

    def get_adadelta_trainer(self, X_sym, y_sym, params, cost, learning_rate,
                             adagrad_param, adadelta_param):
        """ Returns an Adadelta (Zeiler 2012) trainer."""
        raise ValueError("Adadelta trainer is not yet complete!")
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, params)
        self.accumulated_gradients_ = []
        self.accumulated_deltas_ = []
        accumulated_gradients_ = self.accumulated_gradients_
        accumulated_deltas_ = self.accumulated_deltas_

        for layer in self.layers_:
            accumulated_gradients_.extend([shared_zeros(p.shape.eval(),
                                           'accumulated_gradient')
                                           for p in layer.params])
            accumulated_deltas_.extend([shared_zeros(p.shape.eval(),
                                        'accumulated_gradient')
                                        for p in layer.params])

        # compute list of weights updates
        updates = OrderedDict()
        for agrad, adelta, param, gparam in zip(accumulated_gradients_,
                                                accumulated_deltas_,
                                                params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            ag = adadelta_param * agrad + (1 - adadelta_param) * gparam * gparam
            dx = T.sqrt((adelta + adadelta_param) /
                        (agrad + adagrad_param)) * gparam
            # TODO: Norm clipping
            updates[adelta] = (adadelta_param * adelta +
                               (1 - adadelta_param) * dx * dx)
            updates[agrad] = ag
            updates[param] = param - dx

            train_fn = theano.function(inputs=[X_sym, y_sym],
                                       outputs=cost,
                                       updates=updates)
        return train_fn


class MininetBase(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class Softmax(MininetBase):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input_variable, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                              dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred')
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(MininetBase):
    def __init__(self, input_variable, n_in, n_out, rng, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = in

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if not hasattr(self, "W"):
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if not hasattr(self, "b"):
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(MininetBase, TrainingMixin):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, hidden_layer_sizes=[500], batch_size=100, max_iter=1E3,
                 learning_rate=0.01, l1_reg=0., l2_reg=1E-4, random_seed=None,
                 learning_alg="sgd", adagrad_param=1E-6, adadelta_param=0.9,
                 activation="tanh", model_save_name="saved_model",
                 save_every=100):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type hidden_layer_sizes: list of int
        :param hidden_layer_sizes: number of units per hidden layer,
        the dimension of the space in which the labels lie.

        """
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)

        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.adagrad_param = adagrad_param
        self.adadelta_param = adadelta_param
        self.model_save_name = model_save_name
        self.save_every = save_every
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.hidden_layer_sizes = hidden_layer_sizes
        if activation == "relu":
            def relu(x):
                return x * (x > 1e-6)
            self.activation = relu
        elif activation == "tanh":
            self.activation = T.tanh
        elif activation == "sigmoid":
            self.activation = T.nnet.sigmoid
        else:
            raise ValueError("Value %s not understood for activation"
                             % activation)

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:-1]):
            self.layers_.append(HiddenLayer(rng=self.random_state,
                                            input=input_variable,
                                            n_in=n_in, n_out=n_out,
                                            activation=self.activation))
            input_variable = self.layers_[-1].output
        self.layers_.append(Softmax(input=input_variable,
                                    n_in=layer_sizes[-2],
                                    n_out=layer_sizes[-1]))

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.l1 = 0
        for hl in self.layers_:
            self.l1 += abs(hl.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.l2_sqr = 0.
        for hl in self.layers_:
            self.l2_sqr += (hl.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.layers_[-1].negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.layers_[-1].errors

        self.params = self.layers_[0].params
        for hl in self.layers_[1:]:
            self.params += hl.params
        self.cost = self.negative_log_likelihood(y_sym)
        self.cost += self.l1_reg * self.l1
        self.cost += self.l2_reg * self.l2_sqr

        self.predict_function = theano.function(
            inputs=[X_sym], outputs=self.layers_[-1].y_pred)
        self.loss_function = theano.function(
            inputs=[X_sym, y_sym], outputs=self.negative_log_likelihood(y_sym))

        if self.learning_alg == "sgd":
            self.fit_function = self.get_sgd_trainer(X_sym, y_sym, self.params,
                                                     self.cost,
                                                     self.learning_rate)
        elif self.learning_alg == "adagrad":
            self.fit_function = self.get_adagrad_trainer(X_sym, y_sym,
                                                         self.params,
                                                         self.cost,
                                                         self.learning_rate,
                                                         self.adagrad_param)
        elif self.learning_alg == "adadelta":
            self.fit_function = self.get_adadelta_trainer(X_sym, y_sym,
                                                          self.params,
                                                          self.cost,
                                                          self.learning_rate,
                                                          self.adagrad_param,
                                                          self.adadelta_param)
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)

    def partial_fit(self, X, y):
        return self.fit_function(X, y.astype('int32'))

    def fit(self, X, y, valid_X=None, valid_y=None):
        input_size = X.shape[1]
        output_size = len(np.unique(y))
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        self.training_scores_ = []
        self.validation_scores_ = []
        self.training_loss_ = []
        self.validation_loss_ = []
        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)

        batch_indices = list(range(0, X.shape[0], self.batch_size))
        if X.shape[0] != batch_indices[-1]:
            batch_indices.append(X.shape[0])

        start_time = time.clock()
        itr = 0
        best_validation_score = np.inf
        while (itr < self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            itr += 1
            batch_bounds = zip(batch_indices[:-1], batch_indices[1:])
            self.random_state.shuffle(batch_bounds)
            for start, end in batch_bounds:
                self.partial_fit(X[start:end], y[start:end])
            current_training_score = (self.predict(X) != y).mean()
            self.training_scores_.append(current_training_score)
            current_training_loss = self.loss_function(X, y)
            self.training_loss_.append(current_training_loss)
            # Serialize each save_every iteration
            if (itr % self.save_every) == 0 or (itr == self.max_iter):
                f = file(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()
            if valid_X is not None:
                current_validation_score = (
                    self.predict(valid_X) != valid_y).mean()
                self.validation_scores_.append(current_validation_score)
                current_training_loss = self.loss_function(valid_X, valid_y)
                self.validation_loss_.append(current_training_loss)
                print("Validation score %f" % current_validation_score)
                # if we got the best validation score until now, save
                if current_validation_score < best_validation_score:
                    best_validation_score = current_validation_score
                    f = file(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        end_time = time.clock()
        print("Total training time ran for %.2fm" %
              ((end_time - start_time) / 60.))
        return self

    def predict(self, X):
        return self.predict_function(X)


class RandomMLP(MLP, TrainingMixin):
    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        def f(x): return T.tanh(x)
        def df(x): 1 - T.tanh(x) ** 2.
        def castX(value) : # cast numpy array into floatX
            return theano._asarray(value, dtype=theano.config.floatX)

        # make shared var with numpy array
        def sharedX(value, name=None, borrow=False) :
            return theano.shared(castX(value), name=name, borrow=borrow)

        # make shared var with gaussian random values
        def sharedX_randn(shape, mean, stddev, name=None, borrow=False):
            x = np.random.standard_normal(size=shape) * stddev + mean
            return theano.shared(castX(x), name=name, borrow=borrow)

        # make shared var with constant value
        def sharedX_const(shape, value, name=None, borrow=False):
            x = np.zeros(shape) + value
            return theano.shared(castX(x), name=name, borrow=borrow)

        def sharedXs(*values) : # make shared vars at one go
            return ( sharedX(value) for value in values )

        def predict(probs) : # predict labels using probabilities of labels
            return T.argmax(probs, axis=1)

        def softmax_cross_entropy(probs, labels) : # labels are not one-hot code
            return - T.mean( T.log(probs)[T.arange(labels.shape[0]), T.cast(labels,'int32')] )

        def error(pred_labels,labels) : # estimate prediction errors
            return T.mean(T.neq(pred_labels, labels))

        # encoding weights and bias
        nX, nH1, nH2, nY = 784, 1500, 1500, 10 # network definition
        learning_rate = 0.1
        W1, B1 = sharedX_randn((nX, nH1),0,0.01), sharedX_const((nH1,),0)
        W2, B2 = sharedX_randn((nH1,nH2),0,0.01), sharedX_const((nH2,),0)
        W3, B3 = sharedX_randn((nH2, nY),0,0.01), sharedX_const((nY,),0)

        # decoding weights and bias
        V3, V2 = sharedX_randn((nY, nH2),0,0.1), sharedX_randn((nH2,nH1),0,0.1)

        # feed-forward computation, X - data, Y - labels
        X, Y = X_sym, y_sym

        H1_net = T.dot( X, W1 ) + B1 # net values of hidden units
        H1 = f(H1_net)

        H2_net = T.dot( H1, W2 ) + B2 # net values of hidden units
        H2 = f(H2_net)

        P_net = T.dot( H2, W3 ) + B3
        P = T.nnet.softmax(P_net)

        # cost for classification
        cost = softmax_cross_entropy( P, Y )

        # compute derivatives of parameters
        d_P_net = T.grad( cost, P_net )

        d_B3 = d_P_net.sum(axis=0)
        d_W3 = T.dot( H2.T, d_P_net )

        #d_H2 = T.dot( d_P_net, W3.T ) # for back-prop
        d_H2 = T.dot( d_P_net, V3 ) # for feedback alignment
        d_H2_net = d_H2 * df(H2_net)

        d_B2 = d_H2_net.sum(axis=0)
        d_W2 = T.dot( H1.T, d_H2_net )

        #d_H1 = T.dot( d_H2_net, W2.T ) # for back-prop
        d_H1 = T.dot( d_H2_net, V2 ) # for feedback alignment
        d_H1_net = d_H1 * df(H1_net)

        d_B1 = d_H1_net.sum(axis=0)
        d_W1 = T.dot( X.T, d_H1_net )

        # compile train and test function
        err = error( predict(P), Y ) # estimate prediction errors for testing
        self.predict_function = theano.function([X_sym, y_sym], err)
        self.fit_function = theano.function([X_sym, y_sym], [cost],
            updates=OrderedDict({ # just use SGD without momentum
                W3 : W3 - learning_rate * d_W3,  B3 : B3 - learning_rate * d_B3,
                W2 : W2 - learning_rate * d_W2,  B2 : B2 - learning_rate * d_B2,
                W1 : W1 - learning_rate * d_W1,  B1 : B1 - learning_rate * d_B1   })  )
