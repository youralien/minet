import cPickle
from mininet import load_roland
import sys
import matplotlib.pyplot as plt

datasets = load_roland()
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[1]

f = open(sys.argv[1], 'rb')
classifier = cPickle.load(f)

plt.title("Classification Error (relu)")
plt.plot(classifier.training_scores_, color="steelblue", label="Training")
plt.plot(classifier.validation_scores_, color="darkred", label="Validation")
plt.ylabel("Ratio of incorrect examples (out of 1.0)")
plt.xlabel("Epochs")
plt.legend()
plt.figure()
plt.title("Average Loss (relu)")
plt.plot(classifier.training_loss_, color="steelblue", label="Training")
plt.plot(classifier.validation_loss_, color="darkred", label="Validation")
plt.ylabel("Average Loss (log scale)")
plt.xlabel("Epochs")
plt.legend()
plt.show()
