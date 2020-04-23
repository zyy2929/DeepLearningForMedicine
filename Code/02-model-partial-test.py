
###################################
########## import library #########
###################################

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from skimage import io, color

import time
import os
import pickle

import matplotlib.pyplot as plt


###################################
########### load device ###########
###################################

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


###################################
########### dataloader ############
###################################

class ChestXray(Dataset):

    def __init__(self, csv_file, image_root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.image_path = self.data_frame['image_index']
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        image_filename = self.image_root_dir + self.image_path[index]
        image = io.imread(image_filename, as_gray=True)

        # sample is a dictionary which includes the image and 14 labels
        sample = {}

        # since the input to pre-trained network should have 3 channels
        # we need to pad it with two repeatition
        # Note that we need to transpose it since the input size for ToPILImage() is
        # H*W*C instead of C*H*W!!
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)

        # transform the image if transform is not None
        if self.transform:
            image = self.transform(np.uint8(image))
        # add image into the sample dictionary
        sample['image'] = image

        # get the label for the image
        label_col_names = ['normal', 'pneumonia']

        # to get the label for each column
        # 0 --> negative
        # 1 --> positive
        # 2 --> uncertainty (No Finding has no 2)
        sample['label'] = torch.LongTensor([self.data_frame['label'][index]])

        return sample


###################################
############ load test ############
###################################

test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([364,364]),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_df_path = 'chest_xray_origin/test.csv'

root_dir = 'chest_xray_origin/all/'
bs=10

test_loader = DataLoader(ChestXray(test_df_path, root_dir, transform=test_transform), batch_size=bs, shuffle=False)


###################################
########### load model ############
###################################

resnet152_test = models.resnet152(pretrained=True)
resnet152_test = models.resnet152(num_classes=2)
resnet152_test.load_state_dict(torch.load('best_models/best_model_17.pth'))
model = resnet152_test
model.to(device)

test_weights = test_loader.dataset.data_frame.shape[0] / np.array(test_loader.dataset.data_frame['class'].value_counts())[::-1]
test_weights = torch.FloatTensor(test_weights).to(device)
test_criterion = nn.CrossEntropyLoss(weight=test_weights)

optimizer = optim.Adam(model.parameters())


###################################
########### define test ###########
###################################

def test(model, optimizer, criterion, loader, device):

    model.eval()

    running_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    preds_list = []
    truelabels_list = []
    probas_list = []
    with torch.no_grad():
        for batch_idx, samples in enumerate(loader):

            image = samples['image'].to(device)
            label = samples['label'].squeeze()
            label = torch.tensor(label, dtype=torch.long, device=device)

            output = model(image)
            output_softmax = mysoftmax(output)

            _, preds = torch.max(output, dim = 1)

            loss = test_criterion(output, label)
            running_loss += loss.item()

            total_samples += image.shape[0]
            correct += torch.sum(preds == label).item()


            preds_list.append(preds)
            truelabels_list.append(label)
            probas_list.append(output_softmax)

        test_accuracy = correct / total_samples

        return running_loss / len(loader), test_accuracy, preds_list, truelabels_list, probas_list


###################################
############ run test #############
###################################

history_resnet_test = {"test_loss":[], "test_acc":[], "preds_list":[], "truelabels_list":[], "proabs_list":[]}

test_loss, test_acc, preds_list, truelabels_list, proabs_list= test(model, optimizer, test_criterion, test_loader, device)
history_resnet_test["test_loss"].append(test_loss)
history_resnet_test["test_acc"].append(test_acc)
history_resnet_test["preds_list"].append(preds_list)
history_resnet_test["truelabels_list"].append(truelabels_list)
history_resnet_test["proabs_list"].append(proabs_list)

print('{}: test loss: {:.4f} Acc: {:.4f}'.format('test', test_loss, test_acc))
print()


###################################
########## save results ###########
###################################

with open("history_resnet_test.pkl", "wb") as fout:
    pickle.dump(history_resnet_test, fout)
