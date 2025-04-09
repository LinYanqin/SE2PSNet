import os
import h5py
import numpy as np
from torch import nn
from dataset import *
from network import SpinEchoNet, SpatialAttention, ResidualBlock
import scipy.io as sio

device = torch.device('cpu')

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = SpinEchoNet().to(device)
        self.weight_path = 'params/net.pt'
        if os.path.exists(self.weight_path):
            self.net.load_state_dict(torch.load(self.weight_path, map_location=device))
        self.net.eval()

    def forward(self, input1, input2):
        output = self.net(input1, input2)
        return output

    def detect(self, input_path, save_path):
        f = h5py.File(input_path, 'r')

        data_input = np.array(f['mix'])

        print(f"Data input shape: {data_input.shape}")

        data_input_part1 = data_input.T[:, :7]
        data_input_part2 = data_input.T[:, 7:]

        data_input_part1 = np.array(data_input_part1).reshape(1, 1, 4096, 7).astype('float32')
        data_input_part2 = np.array(data_input_part2).reshape(1, 1, 4096, 1).astype('float32')

        data_input1 = torch.from_numpy(data_input_part1.reshape(1, 1, 4096, 7)).to(device)
        data_input2 = torch.from_numpy(data_input_part2.reshape(1, 1, 4096, 1)).to(device)
        data_input2 = np.transpose(data_input2, (0, 3, 2, 1))

        with torch.no_grad():
            results1, results2, x2 = self.net(data_input1, data_input2)


        results_cpu1 = results1.cpu().detach().numpy()

        sio.savemat(save_path, {'input1': data_input1.reshape(4096, 7).tolist(),
                                'input2': data_input2.reshape(1, 4096).tolist(),
                                'pre': results_cpu1.reshape(1, 4096).tolist(),
                                })

        print("Detect successfully!")

if __name__ == '__main__':
    input_path = 'exp/exp_azithromycin.mat'

    # input_path = 'exp/exp_asarone.mat'

    # input_path = 'exp/exp_estradiol.mat'

    # input_path = 'exp/exp_mixture1.mat'
    # input_path = 'exp/exp_mixture2.mat'
    # input_path = 'exp/exp_mixture3.mat'
    # input_path = 'exp/exp_mixture4.mat'


    detector = Detector()
    detector.detect(input_path, 'predict/pre.mat')