from torch import optim
from torch.utils.data import DataLoader
from dataset import *
from network import *
from loss import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def dataset_build(start, end, file_path_label, file_path_data, batch_size):
    dataset = SpinEchoDataSet(start, end, file_path_label, file_path_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

net = SpinEchoNet().to(device)

opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, eps=0.0005)

def train():
    # 训练模型
    global loss
    epoch = 30
    batch_size = 25
    data_num = 5000
    start=0
    end=5000
    index_add = data_num // batch_size

    file_path_label = './label/label.txt'
    file_path_data = './data/data.txt'


    for i in range(epoch):

        data_loader=dataset_build(start, end, file_path_label, file_path_data, batch_size)
        for index_batch, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output, x1, y1 = net(data, target)

            loss = loss_fun(output.float(), target.float())

            opt.zero_grad()

            loss.backward()

            opt.step()

            print(f'epoch:{i+1}  batch:{index_batch + 0 * index_add}  loss:', loss.item())

        # save model
        if (i+1) % 1 == 0:
            torch.save(net.state_dict(), f'params/net_{i+1}.pt')
            print('success to save')
        else:
            print('fail to save')

        lrScheduler.step(loss)
        lr = opt.param_groups[0]['lr']
        print("epoch={}, lr={}".format(i+1, lr))

    print('Finished Training')

if __name__ == "__main__":
    train()