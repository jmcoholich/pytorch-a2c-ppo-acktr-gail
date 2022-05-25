import torch
from torch import nn
import json
from torch.utils.tensorboard import SummaryWriter
import sys
import matplotlib.pyplot as plt
import numpy as np
import time


device = torch.device('cuda')

# hyper-params
epochs = 1000
hidden_layer_size = 2048

# setup
model = nn.Sequential(nn.Linear(28,hidden_layer_size),
	nn.Tanh(),
	nn.Linear(hidden_layer_size, hidden_layer_size),
	nn.Tanh(),
	nn.Linear(hidden_layer_size, 1)).to(device)
print(next(model.parameters()).is_cuda)

optimizer = torch.optim.Adam(model.parameters())

loss_function = torch.nn.MSELoss() 

# import data
with open('data/minitaur_friction_data.txt', 'r') as file:
	data = json.load(file)

keys = list(data)
num_datapoints_per_key = len(data[keys[0]]) 
num_keys = len(data)

assert num_datapoints_per_key * num_keys %10 == 0 
train_size = int(num_datapoints_per_key * num_keys * 9/10)
test_size = int(num_datapoints_per_key * num_keys /10)

X_train = torch.empty((train_size, 28))
Y_train = torch.empty((train_size, 1))

X_test = torch.empty((test_size, 28))
Y_test = torch.empty((test_size, 1))

for i in range(num_keys):
	X_train[i* int(num_datapoints_per_key*0.9): (i + 1) * int(num_datapoints_per_key*0.9)] = torch.Tensor(data[keys[i]])[0: int(0.9 * num_datapoints_per_key)]
	Y_train[i*int(num_datapoints_per_key*0.9): (i + 1) * int(num_datapoints_per_key*0.9)] = eval(keys[i])
	X_test[i* int(num_datapoints_per_key*0.1): (i + 1) * int(num_datapoints_per_key*0.1)] = torch.Tensor(data[keys[i]])[int(0.9 * num_datapoints_per_key): num_datapoints_per_key]
	Y_test[i*int(num_datapoints_per_key*0.1): (i + 1) * int(num_datapoints_per_key*0.1)] = eval(keys[i])

# now normalize data based on training data statistics
x_min, _  = X_train.min(0)
x_max, _ = X_train.max(0)
y_min, _  = Y_train.min(0)
y_max, _ = Y_train.max(0)

X_train = (X_train - x_min) / (x_max - x_min)
Y_train = (Y_train - y_min) / (y_max - y_min)

X_test = (X_test - x_min) / (x_max - x_min)
Y_test = (Y_test - y_min) / (y_max - y_min)


X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device) 
Y_test = Y_test.to(device) 
# breakpoint()
# print(sys.getsizeof(X_train))
print('asd', X_train.is_cuda)

# train
for i in range(epochs):
	start = time.time()
	optimizer.zero_grad()
	predicted_Y = model(X_train)
	loss = loss_function(predicted_Y, Y_train)
	loss.backward()
	optimizer.step()

	predicted_Y = model(X_test)
	test_loss = loss_function(predicted_Y, Y_test)
	print('epoch %d empirical loss value: %0.4f' %(i + 1, loss))
	print('\ttest set loss: %0.4f' %test_loss )
	# breakpoint()
	print('\tEpoch elapsed time: %.2f' % (time.time() - start) + 's')

# get statistics

mean_errors = np.empty((num_keys,))
variances = np.empty((num_keys,))

for i in range(num_keys):
	y_pred = model(X_test[i * int(num_datapoints_per_key * 0.1): (i + 1) * int(num_datapoints_per_key * 0.1)])

	mean_errors[i] = (y_pred.mean() - eval(keys[i])).item()
	variances[i] = y_pred.var().item()  

plt.figure()
plt.subplot(211)
plt.plot([float(i) for i in keys], mean_errors)
plt.grid()
# ax = plt.gca()
# for label in ax.get_xaxis().get_ticklabels()[::10]:
#     label.set_visible(False)

plt.subplot(212)
plt.plot([float(i) for i in keys], variances)
plt.grid()
# ax = plt.gca()
# for label in ax.get_xaxis().get_ticklabels()[::10]:
#     label.set_visible(False)

plt.show()







