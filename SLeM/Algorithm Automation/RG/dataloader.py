import pandas as pd
import jittor as jt
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression,fetch_california_housing,load_diabetes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def make_dataset(name):
    if name == 'abalone':
        data = pd.read_csv(r'./dataset/abalone_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'bodyfat':
        data = pd.read_csv(r'./dataset/bodyfat_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    elif name == 'cpusmall':
        data = pd.read_csv(r'./dataset/cpusmall_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
        feature_number = X.shape[1]
    elif name == 'eunite':
        data = pd.read_csv(r'./dataset/eunite2001.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y_train = data[:, -1]
        x_train = data[:, :-1]

        data = pd.read_csv(r'./dataset/eunite2001.t.csv')
        data = data.values
        y_test = data[:, -1]
        x_test = data[:, :-1]
    elif name == 'housing':
        data = pd.read_csv(r'./dataset/housing_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'mg':
        data = pd.read_csv(r'./dataset/mg_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'mpg':
        data = pd.read_csv(r'./dataset/mpg_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'pyrim':
        data = pd.read_csv(r'./dataset/pyrim_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'space':
        data = pd.read_csv(r'./dataset/space_ga_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'triazines':
        data = pd.read_csv(r'./dataset/triazines_scale.txt.csv')
        data = data.values
        feature_number = int(data.shape[1] - 1)
        y = data[:, -1]
        X = data[:, :-1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    elif name == 'california':
        X, y = fetch_california_housing(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
        feature_number = X.shape[1]
    else:
        print('Dataset Error: Please Prepare your Dataset!!!')



    return x_train, y_train, x_test, y_test, feature_number

def load_data(dataset):
    x_train, y_train, x_test, y_test, feature_number = make_dataset(dataset)

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


    # x_test = torch.FloatTensor(x_test)
    # y_test = torch.FloatTensor(y_test)
    # y_test = torch.unsqueeze(y_test, dim=1)
    # x_train = torch.FloatTensor(x_train)
    # y_train = torch.FloatTensor(y_train)
    # y_train = torch.unsqueeze(y_train, dim=1)

    x_test = torch.FloatTensor(x_test).numpy()
    y_test = torch.FloatTensor(y_test)
    y_test = torch.unsqueeze(y_test, dim=1).numpy()
    x_train = torch.FloatTensor(x_train).numpy()
    y_train = torch.FloatTensor(y_train)
    y_train = torch.unsqueeze(y_train, dim=1).numpy()

    return x_train, y_train, x_test, y_test, feature_number

# x_train, y_train, x_test, y_test, feature_number = load_data('abalone')