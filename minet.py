try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gzip
import tempfile
import os

import numpy as np
import time
import theano
import theano.tensor as T
# Sandbox?
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict


def dropout(random_state, X, keep_prob=0.5):
    if keep_prob > 0. and keep_prob < 1.:
        seed = random_state.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.binomial(n=1, p=keep_prob, size=X.shape,
                             dtype=theano.config.floatX)
        return X * mask
    return X


def fast_dropout(random_state, X):
    seed = random_state.randint(2 ** 30)
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
        try:
            import urllib
            urllib.urlretrieve()
        except AttributeError():
            import urllib.request as urllib

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
        try:
            import urllib
            urllib.urlretrieve()
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    f.close()

    test_set_x, test_set_y = test_set
    test_set_x = test_set_x.astype('float32')
    test_set_y = test_set_y.astype('int32')
    valid_set_x, valid_set_y = valid_set
    valid_set_x = valid_set_x.astype('float32')
    valid_set_y = valid_set_y.astype('int32')
    train_set_x, train_set_y = train_set
    train_set_x = train_set_x.astype('float32')
    train_set_y = train_set_y.astype('int32')

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data(dataset):
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


class MinetBase(object):
    def __init__(self, hidden_layer_sizes, batch_size, max_iter,
                 random_seed, save_frequency, model_save_name):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.max_iter = max_iter
        self.save_frequency = save_frequency
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.model_save_name = model_save_name

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
        self.dropout_layers_ = []
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
            batch_bounds = list(zip(batch_indices[:-1], batch_indices[1:]))
            # Random minibatches
            self.random_state.shuffle(batch_bounds)
            for start, end in batch_bounds:
                self.partial_fit(X[start:end], y[start:end])
            current_training_score = (self.predict(X) != y).mean()
            self.training_scores_.append(current_training_score)
            current_training_loss = self.loss_function(X, y)
            self.training_loss_.append(current_training_loss)
            # Serialize each save_frequency iteration
            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
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
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        end_time = time.clock()
        print("Total training time ran for %.2fm" %
              ((end_time - start_time) / 60.))
        return self

    def predict(self, X):
        return self.predict_function(X)


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


class Softmax(MinetBase):
    def __init__(self, input_variable, n_in=None, n_out=None, weights=None,
                 biases=None):
        if weights is None:
            assert n_in is not None
            assert n_out is not None
            W = theano.shared(value=np.zeros((n_in, n_out),
                                                dtype=theano.config.floatX),
                                name='W', borrow=True)
            b = theano.shared(value=np.zeros((n_out,),
                                                dtype=theano.config.floatX),
                                name='b', borrow=True)
        else:
            W = weights
            b = biases

        self.W = W
        self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input_variable, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred')
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(MinetBase):
    def __init__(self, input_variable, rng, n_in=None, n_out=None, weights=None,
                 biases=None, activation=T.tanh):
        self.input_variable = input_variable
        if not weights:
            assert n_in is not None
            assert n_out is not None
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            W = weights
            b = biases

        self.W = W
        self.b = b

        linear_output = T.dot(self.input_variable, self.W) + self.b
        self.output = (linear_output if activation is None
                       else activation(linear_output))
        self.params = [self.W, self.b]


class MLP(MinetBase, TrainingMixin):
    def __init__(self, hidden_layer_sizes=[500], batch_size=100, max_iter=1E3,
                 dropout=True, learning_rate=0.01, l1_reg=0., l2_reg=1E-4,
                 learning_alg="sgd", adagrad_param=1E-6, adadelta_param=0.9,
                 activation="tanh", model_save_name="saved_model",
                 save_frequency=100, random_seed=None):

        super(MLP, self).__init__(hidden_layer_sizes=hidden_layer_sizes,
                                  batch_size=batch_size, max_iter=max_iter,
                                  random_seed=random_seed,
                                  save_frequency=save_frequency,
                                  model_save_name=model_save_name)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.adagrad_param = adagrad_param
        self.adadelta_param = adadelta_param
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
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
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:-1])):
            keep_prob = 0.8 if i == 0 else 0.5
            if self.dropout:
                keep_prob = 1.0
            self.layers_.append(HiddenLayer(
                rng=self.random_state,
                input_variable=keep_prob * input_variable,
                n_in=n_in, n_out=n_out,
                activation=self.activation))

            dropout_input_variable = dropout(self.random_state, input_variable,
                                             keep_prob=keep_prob)
            W, b = self.layers_[-1].params
            self.dropout_layers_.append(HiddenLayer(
                rng=self.random_state,
                input_variable=dropout_input_variable,
                weights=W, biases=b,
                activation=self.activation))

            input_variable = self.layers_[-1].output

        keep_prob = 0.5
        if self.dropout:
            keep_prob = 1.0
        self.layers_.append(Softmax(input_variable=keep_prob * input_variable,
                                    n_in=layer_sizes[-2],
                                    n_out=layer_sizes[-1]))

        dropout_input_variable = dropout(self.random_state, input_variable,
                                         keep_prob=keep_prob)
        W, b = self.layers_[-1].params
        self.dropout_layers_.append(Softmax(input_variable=dropout_input_variable,
                                    weights=W, biases=b,
                                    n_out=layer_sizes[-1]))

        self.l1 = 0
        for hl in self.layers_:
            self.l1 += abs(hl.W).sum()

        self.l2_sqr = 0.
        for hl in self.layers_:
            self.l2_sqr += (hl.W ** 2).sum()

        self.negative_log_likelihood = self.dropout_layers_[-1].negative_log_likelihood

        self.params = self.layers_[0].params
        for hl in self.layers_[1:]:
            self.params += hl.params
        self.cost = self.negative_log_likelihood(y_sym)
        self.cost += self.l1_reg * self.l1
        self.cost += self.l2_reg * self.l2_sqr

        self.errors = self.layers_[-1].errors
        self.loss_function = theano.function(
            inputs=[X_sym, y_sym], outputs=self.negative_log_likelihood(y_sym))

        self.predict_function = theano.function(
            inputs=[X_sym], outputs=self.layers_[-1].y_pred)

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
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)
