import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from mobileNetv2 import get_model_mobileNetv2
from mobilenet import get_model_mobileNet
from shuffleNet import get_model_shuffleNet
from shuffleNetv2 import get_model_shuffleNet2
from squeeze_net import SqueezeNet

PATH = '/home/xiaoguojian/theoreticalAcademic/'
rgbd_train_data = 'rgbd_seq_train2.txt'
rgbd_test_data = 'rgbd_seq_test2.txt'
rgb_train_data = 'rgb_seq_train2.txt'
rgb_test_data = 'rgb_seq_test2.txt'


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 50


def default_loader(path):
    return np.load(path)


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            # print(len(words))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


train_transforms = transforms.Compose([
    # transforms.ToTensor(),
])

train_data = MyDataset(txt=PATH + rgbd_train_data, transform=train_transforms)
test_data = MyDataset(txt=PATH + rgbd_test_data, transform=train_transforms)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print('train dataset len: {}'.format(len(train_dataloader.dataset)))
print('test dataset len: {}'.format(len(test_dataloader.dataset)))

squueze = SqueezeNet(version=1.1, sample_size=256, sample_duration=32, num_classes=20)
shuffle = get_model_shuffleNet(groups=3, num_classes=20, width_mult=1)
shufflev2 = get_model_shuffleNet2(num_classes=20, sample_size=256, width_mult=1.)
mobile = get_model_mobileNet(num_classes=20, sample_size=256, width_mult=1.)
mobilev2 = get_model_mobileNetv2(num_classes=20, sample_size=256, width_mult=1.)
models = [squueze,shuffle, shufflev2,  mobile, mobilev2]
MODULE = [ 'squeezeNet', 'shuffleNet', 'shuffleNetv2','mobileNet', 'mobileNetv2']
# models = [ mobilev2]

for index, model in enumerate(models):
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)

    total_step = len(train_dataloader)
    loss_data = []
    correct_data = []
    for epoch in range(NUM_EPOCHS):
        loss_data_temp = []
        correct_data_temp = []
        model.train()
        for i, (videos, labels) in enumerate(train_dataloader):

            video = videos.type(torch.FloatTensor).to(DEVICE)
            label = labels.type(torch.FloatTensor).to(DEVICE)
            # print(video.shape)
            outputs = model(video)
            loss = criterion(outputs, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if (i + 1) % 20 == 0:
                loss_data_temp.append(loss.item())
                print('Epoch: [{}/{}],Step: [{}/{}],Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, total_step,
                                                                         loss.item()))
        model.eval()
        with torch.no_grad():
            classes = (
                'c', 'comeon', 's', 'forbid', 'good', 'non', 'dislike', 'helpless', 'come', 'no', 'please', 'pull',
                'push', 'me', 'circle', 'pat', 'wave', 'pray', 'grasp2', 'grasp1')
            classes_correct = [0 for i in range(20)]
            classes_total = [0 for i in range(20)]
            res_predict = []
            res_label = []
            for videos, labels in test_dataloader:
                video = videos.type(torch.FloatTensor).to(DEVICE)
                label = labels.type(torch.FloatTensor).to(DEVICE)
                outputs = model(video)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == label).squeeze()
                for label_idx in range(len(labels)):
                    label_single = int(label[label_idx].item())
                    classes_correct[label_single] += (predicted[label_idx].item() == label[label_idx].item())
                    classes_total[label_single] += 1
                    res_predict.append(predicted[label_idx].item())
                    res_label.append(label_single)

            correct_data_temp = []
            for i in range(20):
                correct_data_temp.append(100 * classes_correct[i] / classes_total[i])
            correct_data_temp.append(100 * sum(classes_correct) / sum(classes_total))
            print('total Accuracy %.4f %%' % (
                    100 * sum(classes_correct) / sum(classes_total)))
            correct_data_temp.append(precision_score(res_label, res_predict, average='macro'))
            correct_data_temp.append(recall_score(res_label, res_predict, average='macro'))
            correct_data_temp.append(f1_score(res_label, res_predict, average='macro'))
            correct_data.append(correct_data_temp)
        loss_data.append(loss_data_temp)

    temp2 = pd.DataFrame(correct_data)
    temp2.columns = ['c', 'comeon', 's', 'forbid', 'good', 'non', 'dislike', 'helpless', 'come', 'no', 'please', 'pull',
                     'push', 'me', 'circle', 'pat', 'wave', 'pray', 'grasp2', 'grasp1', 'total', 'precision_score',
                     'recall_score', 'f1_score']
    temp2.to_excel(MODULE[index] + "correct.xlsx")
    temp = pd.DataFrame(loss_data)
    temp.to_excel(MODULE[index] + "loss.xlsx")
    torch.save(model.state_dict(), MODULE[index] + 'model.kpl')
