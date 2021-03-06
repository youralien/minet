from minet import load_mnist, MLP

datasets = load_mnist()

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')
# construct the MLP class
classifier = MLP(hidden_layer_sizes=[500], random_seed=1999)

print('... training')
classifier.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)
