import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        # model
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3,
                               stride=1, padding=1)
        self.maxpooling1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3,
                               stride=1, padding=1)
        self.maxpooling2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.cx = torch.zeros(1, 8)
        self.hx = torch.zeros(1, 8)
        self.lstm_cell = nn.LSTMCell(input_size=14, hidden_size=8)

    def forward(self, env_info):
        q_table = np.append(env_info, env_info[:3])
        q_table = torch.Tensor(q_table)
        q_table = torch.unsqueeze(q_table, 0)
        q_table = torch.unsqueeze(q_table, 0)
        q_table = self.conv1(q_table)
        q_table = self.maxpooling1(q_table)
        q_table = self.conv2(q_table)
        q_table = self.maxpooling2(q_table)
        q_table = self.flatten(q_table)
        self.hx, self.cx = self.lstm_cell(q_table, (self.hx, self.cx))
        '''
        print state of LSTM cell
        print('hx= ', end='')
        print(self.hx)
        print('cx= ', end='')
        print(self.cx)
        '''
        q_table = self.hx

        return q_table


if __name__ == '__main__':
    Net = LSTM()
    print('Net: ')
    print(Net)
    print()

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(Net.parameters(), lr=0.1)

    for epoch in range(100):

        env_info_1 = [1, 1.414, 1, 5, 7, 19, 88, 7]
        norm_env_info_1 = [i * 0.01 for i in env_info_1]
        q_table_1 = Net(norm_env_info_1)

        env_info_2 = [9, 7, 4, 90, 34, 79, 88, 7.44312]
        norm_env_info_2 = [i * 0.01 for i in env_info_2]
        q_table_2 = Net(norm_env_info_2)

        env_info_experience = torch.tensor([norm_env_info_1, norm_env_info_2])
        q_table_experience = torch.cat((q_table_1, q_table_2), 0)

        label_1 = torch.Tensor([[1, -0.5, -0.3, -0.4, 0.5, 0.8, -0.2, 0.4]])
        label_2 = torch.Tensor([[0.2, -0.5, 0.7, -0.3, 0.5, 0.8, -0.875, 0.4]])
        label = torch.cat((label_1, label_2), 0)

        '''
        print experience and table
        print('env_info_experience: ', end='')
        print(env_info_experience, end='\n\n')
        print('q_table_experience: ', end='')
        print(q_table_experience, end='\n\n')
        print('label: ', end='')
        print(label, end='\n\n')
        '''

        loss = loss_fn(q_table_experience, label)  # compute loss for every net
        opt.zero_grad()  # clear gradients for next train
        loss.backward(retain_graph=True)  # back-propagation, compute gradients
        opt.step()  # apply gradients

        if epoch % 5 == 0:
            print("Q-table: ", end='')
            print(q_table_1)
            print('label:', end='')
            print(label_1, end='\n\n')

            print("Q-table: ", end='')
            print(q_table_2)
            print('label:', end='')
            print(label_2, end='\n\n')

            print('loss=', end='')
            print(loss, end='\n\n')



