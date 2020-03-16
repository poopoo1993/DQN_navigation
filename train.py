from func.readTxtMap import read_map
from func.draw import show
from classes.DQN import LSTM
from classes.robot import Robot
from random import *
import torch
from torch import nn


# initialize variable
mat_map = read_map('map/map')
mobile_bot = Robot()
training_times = 1000
step_limitation = 2000
q_learning_rate = 0.3
q_discount_factor = 0.8


# behavior in every round
def end_point_visible_for(_robot: Robot, _map):

    radar = tuple(_robot.path[-2])
    dx = _robot.end_point[0] - _robot.position[0]
    dy = _robot.end_point[1] - _robot.position[1]
    if abs(dx) > abs(dy):
        for i in range(abs(dx)):
            if abs(dy/dx) * i - abs(radar[1] - _robot.position[1]) < 1:
                radar = (radar[0] + int(abs(dx) / dx), radar[1])
                if _map[radar] == -1:
                    return False
            else:
                radar = (radar[0] + int(abs(dx) / dx), radar[1])
                if _map[radar] == -1:
                    return False
                radar = (radar[0], radar[1] + int(abs(dy) / dy))
                if _map[radar] == -1:
                    return False
    else:
        for i in range(abs(dy)):
            if abs(dx/dy) * i - abs(radar[0] - _robot.position[0]) < 1:
                radar = (radar[0], radar[1] + int(abs(dy) / dy))
                if _map[radar] == -1:
                    return False
            else:
                radar = (radar[0], radar[1] + int(abs(dy) / dy))
                if _map[radar] == -1:
                    return False
                radar = (radar[0] + int(abs(dx) / dx), radar[1])
                if _map[radar] == -1:
                    return False
    return True


def reward(_robot: Robot, _map):

    bonus = 0
    '''
    dx_square = pow((_robot.end_point[0] - _robot.position[0]), 2)
    dy_square = pow((_robot.end_point[1] - _robot.position[1]), 2)
    distance_to_end = pow((dx_square + dy_square), 1/2)
    '''

    if end_point_visible_for(_robot, _map):
        bonus = 0.02
        
    if _map[_robot.position] == -1:
        return -0.1 + bonus
    elif len(_robot.path) > step_limitation:
        return -0.08 + bonus
    elif not _robot.position == _robot.end_point:
        return -0.03 + bonus
    elif _robot.position == _robot.end_point:
        return 1


def q_table_update(_q_table_record, _experience, action_record):
    for i in range(len(_q_table_record) - 2, -1, -1):
        action = action_record[i]
        max_index = int(torch.argmax(_q_table_record[i + 1], dim=1))
        update = q_discount_factor * _q_table_record[i + 1][0][max_index]
        _experience[i][0][action] = _experience[i][0][action] + q_learning_rate * update


def action_in_round(_robot: Robot, _map, _target_networks):

    action_record = list()
    _total_reward = 0
    while True:
        # acting
        _range = _robot.range_finder(_map)
        q_table = mobile_bot.core(_range)
        tn_output = _target_networks(_range)
        selector = randint(0, 9)
        if selector == 0:
            action = randint(0, 7)
            _robot.walk(action)
            action_record.append(action)
        else:
            action = torch.argmax(q_table, dim=1)
            action = int(action[0])
            _robot.walk(action)
            action_record.append(action)

        # Calculate reward
        _reward = reward(_robot, _map)
        _total_reward += _reward

        # Use target network as experience to make update more stable
        tn_output[-1][action] += float(_reward)

        # save experience
        _robot.q_table_record.append(q_table)
        _robot.experience.append(tn_output)

        '''
        # fix Q-table
        _robot.experience[-1][0][action] = (1 - q_learning_rate) * _robot.experience[-1][0][action]
        _robot.experience[-1][0][action] = _robot.experience[-1][0][action] + q_learning_rate * _reward
        _robot.experience[-1][0] = _robot.experience[-1][0].detach()
        '''

        if _map[_robot.position] == -1:
            # q_table_update(_robot.q_table_record, _robot.experience, action_record)
            break
        elif len(_robot.path) > step_limitation:
            # q_table_update(_robot.q_table_record, _robot.experience, action_record)
            break
        elif _robot.position == _robot.end_point:
            # _robot.experience[-1][0][action] = _robot.experience[-1][0][action] + 1/_robot.length_path
            # q_table_update(_robot.q_table_record, _robot.experience, action_record)
            break
        else:
            continue

    return _total_reward


# training
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(mobile_bot.core.parameters(), lr=0.4)
target_networks = LSTM()
# print(mobile_bot.core)
for epoch in range(training_times):

    # find best experience
    # execute one time
    total_reward = action_in_round(mobile_bot, mat_map, target_networks)
    best_record = total_reward
    best_q_table_record = torch.stack(mobile_bot.q_table_record)
    best_experience = torch.stack(mobile_bot.experience)
    mobile_bot.reset()

    for rounds in range(4):
        # start a round and get reward in whole round
        total_reward = action_in_round(mobile_bot, mat_map, target_networks)
        if total_reward > best_record:
            # Save best experience
            best_record = total_reward
            # adjust variable from tensor of tensor to tensor
            best_q_table_record = torch.stack(mobile_bot.q_table_record)
            best_experience = torch.stack(mobile_bot.experience)
            show(mat_map, mobile_bot.path)  
            mobile_bot.reset()
        else:
            mobile_bot.reset()

    # optimize loss
    print(best_experience)
    print(best_q_table_record)
    loss = loss_fn(best_q_table_record, best_experience)  # compute loss
    opt.zero_grad()  # clear gradients for next train
    loss.backward(retain_graph=True)  # back-propagation, compute gradients
    opt.step()  # apply gradients
    print('training epoch #' + str(epoch), end=', ')
    print('loss= ', end='')
    print(float(loss), end=', ')
    print('reward= ', end='')
    print(best_record)

    if epoch % 10 == 0:
        target_networks.load_state_dict(mobile_bot.core.state_dict())


