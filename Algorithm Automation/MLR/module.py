import numpy as np
import math
import jittor as jt
import jittor.nn as nn

# import jittor.Var as Variable
import itertools
import jittor.init as init


class LSTMCell(nn.Module):

    def __init__(self, num_inputs, hidden_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = nn.Linear(num_inputs, 4 * hidden_size)
        self.fc_h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def init_weights(self):
        initrange = 0.1
        self.fc_h2h.weight.uniform_(-initrange, initrange)
        self.fc_i2h.weight.uniform_(-initrange, initrange)

    def execute(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)

        in_gate = nn.Sigmoid()(gates[0])
        forget_gate = nn.Sigmoid()(gates[1] - 1)
        out_gate = nn.Sigmoid()(gates[2])
        in_transform = jt.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * jt.tanh(cx)
        return hx, cx


class MLRSNetCell(nn.Module):

    def __init__(self, num_inputs, hidden_size):
        super(MLRSNetCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = nn.Sequential(
            nn.Linear(num_inputs, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4 * hidden_size)
                                    )
        self.fc_h2h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4 * hidden_size)
                                    )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for module in self.fc_h2h:
            if type(module) == nn.Linear:
                module.weight.uniform_(-initrange, initrange)
        for module in self.fc_i2h:
            if type(module) == nn.Linear:
                module.weight.uniform_(-initrange, initrange)

    def execute(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)

        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)
        # print('gata:', inputs.shape, hx.shape, i2h.shape, h2h.shape, x.shape)

        # print('gata:', len(gates), self.hidden_size)
        # print('gata:', gates[0])
        # print('gata:', gates[1] - 1, (gates[1] - 1))
        # print('gata:', gates[2])
        # print('gata:', gates[3])

        in_gate = nn.Sigmoid()(gates[0])
        forget_gate = nn.Sigmoid()((gates[1] - 1))
        out_gate = nn.Sigmoid()(gates[2])
        in_transform = jt.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * jt.tanh(cx)
        return hx, cx


class MLRSNet(nn.Module):

    def __init__(self, num_layers, hidden_size):
        super(MLRSNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layer1 = MLRSNetCell(1, hidden_size)
        self.layer2 = nn.Sequential(*[MLRSNetCell(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.layer3 = nn.Linear(hidden_size, 1)


    def reset_lstm(self, keep_states=False):

        if keep_states:
            for i in range(len(self.layer2)+1):
                self.hx[i] = jt.Var(self.hx[i].data)
                self.cx[i] = jt.Var(self.cx[i].data)
                self.hx[i], self.cx[i] = self.hx[i], self.cx[i]
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.layer2) + 1):
                self.hx.append(jt.Var(np.zeros((1, self.hidden_size))))
                self.cx.append(jt.Var(np.zeros((1, self.hidden_size))))
                self.hx[i], self.cx[i] = self.hx[i], self.cx[i]


    def execute(self, x):

        if x.size(0) != self.hx[0].size(0):
            self.hx[0] = self.hx[0].expand(x.size(0), self.hx[0].size(1))
            self.cx[0] = self.cx[0].expand(x.size(0), self.cx[0].size(1))
        # print()
        self.hx[0], self.cx[0] = self.layer1(x, (self.hx[0], self.cx[0]))
        x = self.hx[0]

        for i in range(1, self.num_layers):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.layer2[i-1](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.layer3(x)
        out = nn.Sigmoid()(x)
        return out



