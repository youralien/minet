from mininet import load_roland, MLP

datasets = load_roland()

train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[1]

######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')
# construct the MLP class
classifier = MLP(hidden_layer_sizes=[500], random_seed=1999,
                 max_iter=5E3, activation="relu")

print('... training')
classifier.fit(train_set_x, train_set_y, test_set_x, test_set_y)
