from sklearn import datasets
import matplotlib.pyplot as plt 
import numpy as np

iris = datasets.load_iris(return_X_y=False)
X, y = iris['data'], iris['target']

class_color = {0: 'green', 1:'blue', 2:'red'}

separated = {}
for i in range(len(y)):
  if y[i] not in separated:
    separated[y[i]] = []
  separated[y[i]].append(X[i])

figure, axes = plt.subplots(4, 4, figsize=(15,15), constrained_layout=True)
for i in range(4):
  for j in range(4):
    for classValue in separated:
      axes[i, j].scatter(np.array(separated[classValue])[:, j], np.array(separated[classValue])[:, i], color=class_color[classValue], marker='o', s=30)
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_acc(x, y, w):
  y_pred_vals = np.asarray([np.dot(w, row) for row in x])
  y_pred = np.asarray([[activation(y_pred_vals[i][j]) for j in range(len(y_pred_vals[i]))] for i in range(len(y_pred_vals))])
  mult_ones = np.sum(y_pred, axis=1)
  acc = np.zeros(len(y))
  for i in range(len(y)):
    if mult_ones[i] == 1:
      if y_pred[i][y[i]] == 1:
        acc[i] = 1
  return 100*np.mean(acc)  

def activation(s):
  return 1 if s >= 0 else 0

def train_slp(x, y, l):
  w = [np.random.uniform(-0.3, 0.3) for i in range(x.shape[1])]
  epoch = 100
  for epo in range(epoch):    
    for i in range(x.shape[0]):
      o = activation(np.sum(x[i]*w))
      w = w + l*(y[i]-o)*x[i]
  return w

def run_sim(X, y, test_size, sim=10, store=False):
  X = np.column_stack(([1 for i in range(X.shape[0])], X))

  tot_train_acc, tot_test_acc = 0.0, 0.0

  if store:
    dtset_dict = {'Sim':[], 'Train accuracy': [], 'Test accuracy':[]}

  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  y_train_vec = np.transpose(np.asarray([y_train == 0, y_train == 1, y_train == 2], dtype=int))

  for i in range(sim):
    w = np.asarray([train_slp(x_train, y_train_vec[:, i], 0.1) for i in range(3)])

    test_acc = calc_acc(x_test, y_test, w)
    train_acc = calc_acc(x_train, y_train, w)

    if store:
      dtset_dict['Sim'].append(i+1)
      dtset_dict['Train accuracy'].append(train_acc)
      dtset_dict['Test accuracy'].append(test_acc)

      print('Simulation %d: Training acc: %0.2f %%, Testing acc: %0.2f %%' %(i+1, train_acc, test_acc))

    tot_train_acc += train_acc
    tot_test_acc += test_acc

  if store:
    dtset = pd.DataFrame(dtset_dict)
    display(dtset)
    dtset.to_excel('slp_results.xlsx')

  return (tot_train_acc/float(sim), tot_test_acc/float(sim))



iris = datasets.load_iris(return_X_y=False)
X, y = iris['data'], iris['target']

run_sim(X, y, test_size=0.9, store=True)

dt_dict = dict()
dt_dict['Amount of randomly selected training data in %'] = [10, 20, 30, 40, 50, 60]
dt_dict['Training accuracy in %'] = []
dt_dict['Testing accuracy in %'] = []

for i in range(6):
  (tr_acc, ts_acc) = run_sim(X, y, test_size=0.9 - i*0.1)
  dt_dict['Training accuracy in %'].append(tr_acc)
  dt_dict['Testing accuracy in %'].append(ts_acc)

dt = pd.DataFrame(dt_dict, index=[1,2,3,4,5,6])
display(dt)
dt.to_excel('slp_Train_Test_accuracy_comp.xlsx')

plt.plot(dt_dict['Amount of randomly selected training data in %'], dt_dict['Training accuracy in %'], marker='o', color='green', label='Training accuracy')
plt.plot(dt_dict['Amount of randomly selected training data in %'], dt_dict['Testing accuracy in %'], marker='o', color='blue', label='Testing accuracy')
plt.xlabel('Trainig dataset size')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.3, 0.2))
plt.show()
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import math


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
  network = list()
  for layer in n_hidden:
    hidden_layer = [{'weights':[0 for i in range(n_inputs + 1)]} for i in range(layer)]
    network.append(hidden_layer)
    n_inputs = layer
  output_layer = [{'weights':[0 for i in range(n_inputs + 1)]} for i in range(n_outputs)]
  network.append(output_layer)
  return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	activation += np.dot(weights[:-1], inputs)
	return activation

# Transfer neuron activation
def sigmoid(activation):
	return 1.0 / (1.0 + math.exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = sigmoid(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def sigmoid_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, x_train, y_train, l_rate, n_epoch, n_outputs):
  for epoch in range(n_epoch):
    for i in range(len(x_train)):
      row = x_train[i]
      outputs = forward_propagate(network, row)
      expected = [0 for i in range(n_outputs)]
      expected[y_train[i]] = 1
      backward_propagate_error(network, expected)
      update_weights(network, row, l_rate)

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 
def run_sim(X, y, n_hidden, test_size, l_rate=0.1, n_epoch=1000, sim=10, store=False):
	tot_train_acc, tot_test_acc = 0.0, 0.0

	if store:
		dtset_dict = {'Sim':[], 'Train accuracy': [], 'Test accuracy':[]}

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)		

	for i in range(sim):
		n_inputs = x_train.shape[1]
		n_outputs = len(set(y_train))

		network = initialize_network(n_inputs, n_hidden, n_outputs)
		train_network(network, x_train, y_train, l_rate, n_epoch, n_outputs)

		y_pred = [predict(network, row) for row in x_test]
		test_acc = metrics.accuracy_score(y_test, y_pred)*100

		y_pred = [predict(network, row) for row in x_train]
		train_acc = metrics.accuracy_score(y_train, y_pred)*100

		if store:
			dtset_dict['Sim'].append(i+1)
			dtset_dict['Train accuracy'].append(train_acc)
			dtset_dict['Test accuracy'].append(test_acc)

			print('Simulation %d: Training acc: %0.2f %%, Testing acc: %0.2f %%' %(i+1, train_acc, test_acc))

		tot_train_acc += train_acc
		tot_test_acc += test_acc

	if store:
		dtset = pd.DataFrame(dtset_dict)
		display(dtset)
		dtset.to_excel('mlp_results.xlsx')

	return (tot_train_acc/float(sim), tot_test_acc/float(sim))
 

iris = datasets.load_iris(return_X_y=False)
X, y = iris['data'], iris['target']

n_hidden = [6, 6]

run_sim(X, y, n_hidden, test_size=0.9, store=True)

dt_dict = dict()
dt_dict['Amount of randomly selected training data in %'] = [10, 20, 30, 40, 50, 60]
dt_dict['Training accuracy in %'] = []
dt_dict['Testing accuracy in %'] = []

for i in range(6):
	print('Test size is: '+str(i))
	(tr_acc, ts_acc) = run_sim(X, y, n_hidden, test_size=0.9- 0.1*i)
	dt_dict['Training accuracy in %'].append(tr_acc)
	dt_dict['Testing accuracy in %'].append(ts_acc)

dt = pd.DataFrame(dt_dict, index=[1,2,3,4,5,6])
display(dt)
dt.to_excel('mlp_Train_Test_accuracy_comp.xlsx')

plt.plot(dt_dict['Amount of randomly selected training data in %'], dt_dict['Training accuracy in %'], marker='o', color='green', label='Training accuracy')
plt.plot(dt_dict['Amount of randomly selected training data in %'], dt_dict['Testing accuracy in %'], marker='o', color='blue', label='Testing accuracy')
plt.xlabel('Trainig dataset size')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.3, 0.2))
plt.show()
