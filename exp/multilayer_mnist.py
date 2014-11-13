import os
import time
import numpy as np
from mininet import load_data, MLP

datasets = load_data('mnist.pkl.gz')

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
train_set_x = np.vstack((train_set_x, valid_set_x))
train_set_y = np.concatenate((train_set_y, valid_set_y))

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# construct the MLP class
classifier = MLP(hidden_layer_sizes=[1500, 1500, 1500], learning_rate=0.1,
                 batch_size=20, l2_reg=0., random_seed=1999)

print '... training'
start_time = time.clock()
classifier.fit(train_set_x, train_set_y, test_set_x, test_set_y)
end_time = time.clock()
print('The code for file ' + os.path.split(__file__)[1] +
      ' ran for %.2fm' % ((end_time - start_time) / 60.))
