import numpy
import scipy.special
import matplotlib.pyplot


class neural_network_1:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.i_nodes = input_nodes
		self.h_nodes = hidden_nodes
		self.o_nodes = output_nodes

		self.learning_rate = learning_rate

		# create weights in range from -0.5 to 0.5 (matrices)
		#self.W_ih = numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5
		#self.W_ho = numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5

		# more sophisticated approach to generating weights
		# numpy.random.normal(loc=0.0, scale=1.0, size=None)
		# 	loc - Mean (“centre”) of the distribution
		#	scale - Standard deviation (spread or “width”) of the distribution. Must be non-negative.
		#	size - Output shape.
		# we set the standard deviation of the distribution to 1 / sqrt(number of incoming links)
		self.W_ih = numpy.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
		self.W_ho = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

		# set the activation function
		# in our case we'll use sigmoid function as the activation function, which in scipy.special is called expit(x)
		self.activation_function = lambda x: scipy.special.expit(x)
	
	def train(self, inputs_list, targets_list):
		# convert inputs_list to 2d array
		inputs = numpy.array(inputs_list, ndmin = 2).T
		
		# convert targets_list to 2d array
		targets = numpy.array(targets_list, ndmin = 2).T

		# feed forward ///////////////////////////////
		
		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.W_ih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into output layer
		final_inputs = numpy.dot(self.W_ho, hidden_outputs)
		# calculate the signals emerging from output layer
		final_outputs = self.activation_function(final_inputs)

		# backpropagate the errors ///////////////////////////////

		# get the errors for updating the weights between hidden and output layers
		output_errors = targets - final_outputs

		# get the errors for updating the weights between input and hidden layers
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = numpy.dot(self.W_ho.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.W_ho += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.W_ih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

	# predict (used to predict on real-world data, after training the model)
	def query(self, inputs_list):
		# convert inputs_list to 2d array
		inputs = numpy.array(inputs_list, ndmin = 2).T

		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.W_ih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into output layer
		final_inputs = numpy.dot(self.W_ho, hidden_outputs)
		# calculate the signals emerging from output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs


def main():
	# 28 x 28 pixels image, so 28 * 28 = 784 input nodes
	input_nodes = 784
	hidden_nodes = 100
	output_nodes = 10

	learning_rate = 0.38
	
	nn = neural_network_1(input_nodes, hidden_nodes, output_nodes, learning_rate)

	# Get the train examples
	training_data_file = open("mnist_train_500.csv", 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	# prepare the data and train neural network

	for example in training_data_list:
		all_values = example.split(',')

		# scale the input (change the range of color values from 0 - 255 to 0.01 - 1.0)
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

		# create target output values (all 0.01, except the desired label, which is 0.99)
		targets = numpy.zeros(output_nodes) + 0.01
		# all_values[0] is the target label fro this example
		targets[int(all_values[0])] = 0.99

		nn.train(inputs, targets)
	
	# Get the test examples
	test_data_file = open("mnist_test_50.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	# test neural network

	amount_of_test_examples = 50
	recognised_correctly = 0
	
	for record in test_data_list:
		all_values = record.split(",")

		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		outputs = nn.query(inputs)

		label = numpy.argmax(outputs)

		if (int(all_values[0]) == label):
			recognised_correctly += 1

		print("\nactual answer: ", all_values[0])
		print("neural network's answer: ", label, "(", outputs[label], ")")
	
	print("Accuracy: ", recognised_correctly / amount_of_test_examples)


main()