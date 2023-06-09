import jittor as jt
from jittor import nn, init
import math

# class Linear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         self.in_features = in_features
#         self.out_features = out_features
#         # torch
#         # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         # if self.bias is not None:
#         #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#         #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         #     init.uniform_(self.bias, -bound, bound)
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         self.weight = init.kaiming_uniform_((out_features, in_features), "float32")
#         bound = 1.0/math.sqrt(in_features)
#         self.bias = init.uniform((out_features,), "float32",-bound,bound) if bias else None
#
#     def execute(self, x):
#         x = matmul_transpose(x, self.weight)
#         if self.bias is not None:
#             return x + self.bias
#         return x

class Log_Regression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            init.uniform_(self.linear.bias, -bound, bound)
        self.sigmoid = nn.Sigmoid()
    def execute(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        output = nn.relu(self.hidden(x))
        output = self.sigmoid(self.predict(output))
        return output

class Linear_Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear_Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def execute(self, x):
        return self.linear(x)