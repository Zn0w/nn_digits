import numpy
import scipy.special


# a neural net with 3 layers (input, 1 hidden, output)
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
	# create a neural net with 3 layers (input, 1 hidden, output), 3 neurons in each layer

	nn = neural_network_1(3, 3, 3, 0.2)
	nn.train([1.0, -0.4, 0.5], [0.2, 0.9, 0.35])
	print(nn.query([1.0, 0.8, 0.5]))


main()