import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
import matplotlib.pyplot as plt

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)
X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
# X_data = torch.from_numpy(X_data).to(device)
# Y_data = torch.from_numpy(Y_data).to(device)
X_data = torch.from_numpy(X_data)
Y_data = torch.from_numpy(Y_data)

trX = X_data[:5000]
trY = Y_data[:5000]
tsX = X_data[5000:]
tsY = Y_data[5000:]

washout = [500]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size)
    # model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    output, hidden = model(tsX, [0], hidden)
    print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")

    # Visualize
    Y_test = tsY.detach().numpy()
    ESN_test = output.detach().numpy()
    t = np.linspace(0,1,Y_test.shape[0])
    plt.plot(t,Y_test[:,0,0])
    plt.plot(t,ESN_test[:,0,0])
    plt.show(block=True)