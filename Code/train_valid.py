
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

from sklearn import metrics

from skimage import io, color

import time
import os
import pickle

import matplotlib.pyplot as plt
import scikitplot as skplt


###################################
######### def train/valid #########
###################################

def train(epoch, model, optimizer, criterion, loader, device):

    model.train()

    running_loss = 0.0
    epoch_loss = 0.0
    total_samples = 0
    correct = 0
    mysoftmax = nn.Softmax(dim=1)

    for batch_idx, samples in enumerate(loader):

        image = samples['image'].to(device)
        label = samples['label'].squeeze()
        label = torch.tensor(label, dtype=torch.long, device=device)

        output = model(image)
        _, preds = torch.max(output, dim = 1)

        loss = criterion(output, label)
        running_loss += loss.item()
        epoch_loss += loss.item()

        total_samples += image.shape[0]
        correct += torch.sum(preds == label).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = 0.0

    training_accuracy = correct / total_samples

    return epoch_loss / len(loader), training_accuracy

def validation(epoch, model, optimizer, criterion, loader, device):

    model.eval( )

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

            loss = criterion(output, label)
            running_loss += loss.item()

            total_samples += image.shape[0]
            correct += torch.sum(preds == label).item()

            preds_list.append(preds.cpu().numpy())
            truelabels_list.append(label.cpu().numpy())
            probas_list.append(output_softmax.cpu().numpy())

        valid_accuracy = correct / total_samples

        probas_list = np.vstack(probas_list)
        truelabels_list = np.concatenate(truelabels_list)
        preds_list = np.concatenate(preds_list)

        auc_score = metrics.roc_auc_score(truelabels_list, preds_list)

        return running_loss / len(loader), valid_accuracy, preds_list, truelabels_list, probas_list, auc_score
