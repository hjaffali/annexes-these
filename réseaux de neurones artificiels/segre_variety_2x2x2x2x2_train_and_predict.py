from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import utils
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from random import randint
from scipy.stats import norm
from scipy.stats import ortho_group

import numpy as np
import sys
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#size of the training dataset
training_size = 502600
#size of the validation dataset
validation_size = 55600
#size of the testing dataset
testing_size = 32000
#number of epochs
nbEpochs = 100

#size of a single input vector
input_data_size = 2*2*2*2*2
#size of the output
output_data_size = 1
#number of classes in the classification problem
nb_of_classes = 2


def power_2_activation(x):
	"""
		Square activation function : x -> x^2

		Parameters :
		- x: input number (weighted sum in the context of activation function)

		Return : x^2, the input number squared

		"""
	return (x**2)



def generate_data(size,input_data_size,nb_of_classes):
	"""
		Function used to generate datasets (training, validation and testing)

		Parameters :
		- size : size of the datasets, number of vectors
		- input_data_size : size of each vector (number of entries) of the dataset
		- nb_of_classes : number of classes in the classification problem (used to divide equally the dataset)

		Return :
		- inputs : multidimentional array regrouping all input vectors. The array is of size : size*input_data_size
		- outputs : array regrouping all outputs associated to each input vector. The array is of size : size
	"""

	# initializing inputs and outputs arrays
	inputs = np.empty([size, input_data_size])
	outputs = np.empty([size, 1])

	# subdividing the dataset depending on the number of classes
	temp = int(size/nb_of_classes)
	for i in range(0,temp):
		#rank 1 tensors, separable states
		state = generate_point_on_segre_variety()
		inputs[i] = state.copy()
		outputs[i]= [0]

	for i in range(temp,temp+int((size-temp)/5)):
		#rank 2 tensors
		state = normalize(random_rank_one(5)+random_rank_one(5))
		inputs[i] = state.copy()
		outputs[i]= [1]

	for i in range(temp+int((size-temp)/5),temp+2*int((size-temp)/5)):
		#border rank 3 tensors
		state = normalize(random_rank_one(5)+random_rank_one(5)+random_rank_one(5))
		inputs[i] = state.copy()
		outputs[i]= [1]

	for i in range(temp+2*int((size-temp)/5),temp+3*int((size-temp)/5)):
		#border rank 4 tensors
		state = normalize(random_rank_one(5)+random_rank_one(5)+random_rank_one(5)+random_rank_one(5))
		inputs[i] = state.copy()
		outputs[i]= [1]

	for i in range(temp+3*int((size-temp)/5),temp+4*int((size-temp)/5)):
		#border rank 5 tensors
		state = normalize(random_rank_one(5)+random_rank_one(5)+random_rank_one(5)+random_rank_one(5)+random_rank_one(5))
		inputs[i] = state.copy()
		outputs[i]= [1]

	for i in range(temp+4*int((size-temp)/5),size):
		#random tensors
		state = random_tensor(5)
		inputs[i] = state.copy()
		outputs[i]= [1]

	return inputs, outputs

def random_rank_one(dimension):
	"""
		Generate a rank one tensor

		Parameters :
		- dimension : number of qubits

		Return :
		- vector : normalized rank one tensor of size 2^dimension
	"""
	vector = np.random.random(2)*2-1
	for i in range(0,dimension-1):
		# Kronecker product
		vector = np.kron(vector,np.random.random(2)*2-1)
	vector = normalize(vector)
	return vector

def random_tensor(dimension):
	"""
		Generate a random tensor

		Parameters :
		- dimension : number of qubits

		Return :
		- vector : normalized random tensor of size 2^dimension
	"""
	vector = np.random.random(2**dimension)*2-1
	vector = normalize(vector)
	return vector

def generate_point_on_segre_variety():
	"""
		Generate a 5-qubit separable state

		Return :
		- vector : 5-qubit normalized separable state
	"""
	vector = random_rank_one(5)
	return vector

def generate_SLOCC_2x2():
	"""
		Generate a random 2x2 SLOCC operator

		Return :
		- vec : XXXXXXXX
	"""
	vec = np.random.random((2,2))*2 -1
	det = np.linalg.det(vec)
	while (det<0):
		vec = np.random.random((2,2))*2 -1
		det = np.linalg.det(vec)
	vec = vec/np.sqrt(det)
	return vec

def apply_SLOCC(vector):
	"""
		Act on the vector in parameter with a random element of the SLOCC group

	"""
	slocc1 = generate_SLOCC_2x2()
	slocc2 = generate_SLOCC_2x2()
	slocc3 = generate_SLOCC_2x2()
	slocc4 = generate_SLOCC_2x2()
	slocc5 = generate_SLOCC_2x2()
	matrix = np.kron(np.kron(np.kron(np.kron(slocc1,slocc2),slocc3),slocc4),slocc5)
	return matrix.dot(vector)

def normalize(v):
	"""
		Normalize the vector in parameter

		Parameters :
		- v : the vector to be normalized

		Return :

	"""
	norm=np.linalg.norm(v, ord=1)
	if norm==0:
		return v
	else :
		return v / norm



# define the LeakyReLU activation fuction
leakyRELU = layers.LeakyReLU(alpha=0.4)

# define the neural network model
model = Sequential()

## Input Layer
# Dense(100) is a fully-connected layer with 100 hidden units.
# in the first layer, you must specify the expected input data shape
model.add(Dense(100, input_dim=input_data_size, init='uniform'))
model.add(leakyRELU)

## Hidden Layer
model.add(Dense(50, init='uniform'))
model.add(leakyRELU)

## Hidden Layer
model.add(Dense(25, init='uniform'))
model.add(leakyRELU)

## Hidden Layer
model.add(Dense(16, init='uniform'))
model.add(leakyRELU)

## Output Layer
model.add(Dense(output_data_size, init='uniform'))
model.add(Activation('sigmoid'))


# compiling the model
model.compile(loss='binary_crossentropy',
			optimizer='nadam',
			metrics=['binary_accuracy'])


# generate dummy training data
x_train, y_train = generate_data(training_size,input_data_size,nb_of_classes)
print(x_train,y_train)

# generate dummy validation data
x_validation, y_validation = generate_data(validation_size,input_data_size,nb_of_classes)
print(x_validation,y_validation)

# generate dummy test data
x_test, y_test = generate_data(testing_size,input_data_size,nb_of_classes)
print(x_test,y_test)

# train the neural network model
model.fit(x_train, y_train,
			nb_epoch=nbEpochs,
			batch_size=128,
			verbose=1,
			shuffle=True,
			validation_data=(x_validation, y_validation))

# evaluate the trained model on the training , validation and testing models
loss_train, acc_train = model.evaluate(x_train, y_train, batch_size=128)
loss_validation, acc_validation = model.evaluate(x_validation, y_validation, batch_size=128)
loss_test, acc_test = model.evaluate(x_test, y_test, batch_size=128)
# printing all the losses, accuracies, and the neural network weights
print(loss_train, acc_train)
print(loss_validation, acc_validation)
print(loss_test, acc_test)
print(model.metrics_names)
print(model.get_weights())


# saving the model in a .h5 file
model.save('model_ReLu_point_in_segre_2x2x2x2x2_database_'+str(training_size)+'_validation_'+str(validation_size)+'_test_'+str(testing_size)+'_epoch_'+str(nbEpochs)+'.h5')


######### USING THE MODEL FOR PREDICTIONS #############

#suppose we want to make predictions for the state Phi 5
def generate_state_phi_5():
	vector = np.zeros(2*2*2*2*2)
	vector[0] = 1
	vector[7] = 1
	vector[11] = 1
	vector[13] = 1
	vector[14] = 1
	vector[19] = 1
	vector[21] = 1
	vector[22] = 1
	vector[25] = 1
	vector[26] = 1
	vector[28] = 1
	vector[31] = 1
	vector = normalize(apply_SLOCC(vector))
	return vector

def generate_data_phi_5(size,input_data_size):
	inputs = np.empty([size, input_data_size])
	for i in range(0,size):
		state = generate_state_phi_5()
		inputs[i] = state.copy()
	return inputs

# size of the predicition dataset, which is the number of SLOCC equivalent states used to make the prediction
data_size = 10000

# generate the prediction dataset
prediction_X = generate_data_phi_5(data_size,input_data_size)

# making prediction with the trained network
prediction = model.predict(prediction_X,verbose=1)

print(prediction)


# best fit of data, determining mean and variance
(mu, sigma) = norm.fit(prediction)

# constructing the histogram of the data
n, bins, patches = plt.hist(prediction,bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],align='mid')
print(n,bins,patches)


#plotting the histogram
plt.xlabel('Binary classes')
plt.ylabel('Proportion')
plt.xlim(0, 1)
plt.title(r'$\mathrm{Mean\ value}\  \mu=%.3f,\mathrm{Variance}\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)

plt.show()
