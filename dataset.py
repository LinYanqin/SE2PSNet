import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

class SpinEchoDataSet(Dataset):
    def __init__(self, start, end, file_path_label, file_path_data):
        with open(file_path_label, 'r') as flabel:
            self.labelset = flabel.readlines()[start * 3: end * 3]
        with open(file_path_data, 'r') as fdata:
            self.dataset = fdata.readlines()[start * 8: end * 8]

        self.dataset_size = len(self.dataset) // 8

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        data = self.dataset[(index * 8):((index + 1) * 8)]  # 8*4096, list[str]
        label = self.labelset[(index * 3):((index + 1) * 3)]  # 3*4096, list[str]
        temp_data = [i.strip('\n').split('\t') for i in data]  # 8*4096, list[list[str]]
        temp_label = [i.strip('\n').split('\t') for i in label]  # 3*4096, list[list[str]]
        for i in range(len(temp_data[0])):
            for j in range(len(temp_data)):
                try:
                    temp_data[j][i] = float(temp_data[j][i])  # 8*4096, list[list[int]]
                except ValueError:
                    temp_data[j][i] = 0.0
            for k in range(len(temp_label)):
                try:
                    temp_label[k][i] = float(temp_label[k][i])  # 3*4096, list[list[int]]
                except ValueError:
                    temp_label[k][i] = 0.0

        data = torch.Tensor(temp_data)
        data = data.unsqueeze(-1).transpose(0, 2)  # 1*4096*8, tensor
        label = torch.Tensor(temp_label)
        label = label.unsqueeze(-1)  # 3*4096*1, tensor
        return data, label

    def get_sample_size(self, index):
        sample_data, sample_label = self.__getitem__(index)
        return sample_data.size(), sample_label.size()

    def print_sample(self, index):
        sample_data, sample_label = self.__getitem__(index)
        # print(sample_label[0,...] > 0)  # 查看通道0标签
        data_size = sample_data.size()
        label_size = sample_label.size()

        print(f"Sample {index} Data Size: {data_size}")
        print(f"Sample {index} Data:")
        print(sample_data)

        print(f"Sample {index} Label Size: {label_size}")
        print(f"Sample {index} Label:")
        print(sample_label)

    def plot_sample(self, index):
        sample_data, sample_label = self.__getitem__(index)

        data_np = sample_data.squeeze().numpy().T  # 8*4096
        label_np = sample_label.squeeze().numpy()  # 3*4096
        print(label_np.shape)

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"NMR Signal Data of Sample {index + 1} ", fontsize=16)
        for i in range(data_np.shape[0]):
            plt.subplot(2, 4, i + 1)
            plt.plot(data_np[i])
            plt.xlabel("point")
            plt.ylabel("value")
            plt.title(f"sigal {i + 1}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"NMR Signal Label of Sample {index + 1} ", fontsize=16)
        for i in range(label_np.shape[0]):
            plt.subplot(3, 1, i + 1)
            plt.plot(label_np[i])
            plt.xlabel("point")
            plt.ylabel("value")
            if(i == 0):
                plt.title(f"strength")
            elif(i == 1):
                plt.title(f"chemical shift")
            elif (i == 2):
                plt.title(f"strength with T2")

        plt.tight_layout()
        plt.show()
