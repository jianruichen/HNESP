import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, lap_matrix_p, lap_matrix_n, incidence_matrix_preoder):
        K0 = 0.5
        K1 = 0.05
        K2 = -0.05
        K3 = 0.05
        output2 = torch.zeros_like(input)
        output1 = K0 * input + K1 * torch.sin(torch.mm(lap_matrix_p, input)) + \
                               K2 * torch.sin(torch.mm(lap_matrix_n, input))
        for i in range(input.shape[0]):
            c = 0
            high_structure = np.array(np.where(incidence_matrix_preoder[i, :] == 1))
            for k in range(high_structure.shape[1]):
                structure_list = np.array(np.where(incidence_matrix_preoder[:, high_structure[0, k]] == 1))
                if structure_list.shape[1] == 3:#
                     c = c + F.tanh(2 * input[structure_list[0, 0], :] - (input[structure_list[0, 1], :]) - input[structure_list[0, 2], :])
                elif structure_list.shape[1] == 4:
                     c = c + F.tanh(3 * input[structure_list[0, 0], :] - (input[structure_list[0, 1], :])
                                    - input[structure_list[0, 2], :] - input[structure_list[0, 3], :])
                elif structure_list.shape[1] == 10:
                     c = c + F.tanh(9 * input[structure_list[0, 0], :] - (input[structure_list[0, 1], :])
                                    - input[structure_list[0, 2], :] - input[structure_list[0, 3], :]
                                    - input[structure_list[0, 4], :] - input[structure_list[0, 5], :]
                                    - input[structure_list[0, 6], :] - input[structure_list[0, 7], :]
                                    - input[structure_list[0, 8], :] - input[structure_list[0, 9], :])
            output2[i, :] = K3 * c
        output = output1 + output2
        final_output = torch.mm(output, self.weight)
        if self.bias is not None:
            return final_output + self.bias
        else:
            return final_output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

