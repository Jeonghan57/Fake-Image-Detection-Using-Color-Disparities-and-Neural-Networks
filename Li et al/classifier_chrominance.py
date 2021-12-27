# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:26:55 2021

@author: Jeonghan_lee
"""

import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu 사용 가능?

import pandas as pd


# 파일 경로
location_train = './csv/LSUN/cat/PGGAN/YCbCr'
location_test = './csv/LSUN/church/PGGAN/YCbCr'

channel = 3

param = 75 * channel

# 추출 및 변환 코드
#train_fake
train_fake_pd = pd.read_csv('{}/{}'.format(location_train, 'train_features_fake_all.csv'),
                      header=None, index_col=None, names=None)
train_fake_np = pd.DataFrame.to_numpy(train_fake_pd)
#train_real
train_real_pd = pd.read_csv('{}/{}'.format(location_train, 'train_features_real_all.csv'),
                      header=None, index_col=None, names=None)
train_real_np = pd.DataFrame.to_numpy(train_real_pd)

#test_fake
test_fake_pd = pd.read_csv('{}/{}'.format(location_test, 'test_features_fake.csv'),
                      header=None, index_col=None, names=None)
test_fake_np = pd.DataFrame.to_numpy(test_fake_pd)
#test_real
test_real_pd = pd.read_csv('{}/{}'.format(location_test, 'test_features_real.csv'),
                      header=None, index_col=None, names=None)
test_real_np = pd.DataFrame.to_numpy(test_real_pd)


def get_data(batch_size, train=True):
    if train:
        temp_real = train_real_np[:25000] #20000
        temp_real = np.reshape(temp_real, [25000, param])
        temp_fake = train_fake_np[:25000] #20000
        temp_fake = np.reshape(temp_fake, [25000, param])
    else:
        temp_real = test_real_np[:2000] #2000
        temp_real = np.reshape(temp_real, [temp_real.shape[0], param])
        temp_fake = train_fake_np[:2000] #2000
        temp_fake = np.reshape(temp_fake, [temp_fake.shape[0], param])
    data_holder = np.concatenate((temp_real, temp_fake))
    size_real, size_fake = temp_real.shape[0], temp_fake.shape[0]
    del temp_real, temp_fake
    
    print("\nData Loading Complete")
    labels = np.zeros([size_real + size_fake, 2])
    labels[:size_real, 0] = 1
    labels[size_real:, 1] = 1
    
    data_holder = torch.from_numpy(data_holder).float()
    labels = torch.from_numpy(labels).long()
    ds = TensorDataset(data_holder, labels)
    del data_holder, labels
    data_loader=DataLoader(ds, batch_size=batch_size, shuffle=train)
    return data_loader


def train(Net, batch_size):
    Net = Net.to(device)
    lr = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=lr, momentum=0.9)
    
    print("Training Start")
    
    train_data = get_data(batch_size, True)
    for epoch in range(50):
        count = 0
        for X, Y in train_data:
            X = X.to(device)
            Y = Y.to(device)
            
            y_pred = Net(X)
            
            loss = loss_fn(y_pred, torch.max(Y, 1)[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count +=1
            if count % 100 == 0:
                print(f"epoch:{epoch}, loss={loss}")
        
    torch.save(Net.state_dict(), f"./result/LSUN/epoch(cat(PGGAN)_YCbCr)_{epoch}_1.pth")


def Evaluate_Networks(Net):
    save_path = "./result/LSUN/"
    # data
    Net.load_state_dict(torch.load(save_path + "epoch(cat(PGGAN)_YCbCr)_49_1.pth"), strict=False)
    Net = Net.to(device).eval()
    
    test_data = get_data(64, train=False)
    
    # Test
    ys = []
    ypreds = []
    for X, Y in tqdm.tqdm(test_data):
        X = X.to(device)
        Y = Y.to(device)
        
        with torch.no_grad():
            _, y_pred = Net(X).max(1)
            ys.append(Y.max(1)[1])
            ypreds.append(y_pred)
            
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    
    acc_real = (ys[2000:] == ypreds[2000:]).float().sum() / len(ys[2000:])
    acc_fake = (ys[:2000] == ypreds[:2000]).float().sum() / len(ys[:2000])
    acc = (ys == ypreds).float().sum() / len(ys)
    
    print('\nReal Accuracy : ', acc_real.item())
    print('Fake Accuracy : ', acc_fake.item())
    print('Tatal AVG : ', acc.item())
    
    

net = nn.Linear(225, 2)
# train(net , 64)
Evaluate_Networks(net)