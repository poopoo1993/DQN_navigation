import numpy as np
import torch
from classes.DQN import LSTM


class Robot:

    def __init__(self, _start_point=(10, 10), _end_point=(90, 90), _path=list()):

        # parameter of robot
        self.start_point = _start_point
        self.end_point = _end_point
        self.position = _start_point
        self.path = _path
        self.length_path = 0
        self.path.append(self.position)

        # parameter of neural networks
        self.core = LSTM()
        self.target_network = LSTM()
        self.experience = list()
        self.q_table_record = list()

    def reset(self):

        # reset robot state
        self.position = self.start_point
        self.path = list()
        self.path.append(self.position)
        self.length_path = 0

        # reset memory
        self.experience = list()
        self.q_table_record = list()

    def walk(self, direction):
        if direction == 0:    # walk to top
            self.position = (self.position[0], self.position[1] - 1)
            self.path.append(tuple(self.position))
            self.length_path += 1

        elif direction == 1:  # walk to top_right
            self.position = (self.position[0] + 1, self.position[1] - 1)
            self.path.append(tuple(self.position))
            self.length_path += pow(2, 1/2)

        elif direction == 2:  # walk to right
            self.position = (self.position[0] + 1, self.position[1])
            self.path.append(tuple(self.position))
            self.length_path += 1

        elif direction == 3:  # walk to right_down
            self.position = (self.position[0] + 1, self.position[1] + 1)
            self.path.append(tuple(self.position))
            self.length_path += pow(2, 1/2)

        elif direction == 4:  # walk to down
            self.position = (self.position[0], self.position[1] + 1)
            self.path.append(tuple(self.position))
            self.length_path += 1

        elif direction == 5:  # walk to left_down
            self.position = (self.position[0] - 1, self.position[1] + 1)
            self.path.append(tuple(self.position))
            self.length_path += pow(2, 1/2)

        elif direction == 6:  # walk to left
            self.position = (self.position[0] - 1, self.position[1])
            self.path.append(tuple(self.position))
            self.length_path += 1

        elif direction == 7:  # walk to top_left
            self.position = (self.position[0] - 1, self.position[1] - 1)
            self.path.append(tuple(self.position))
            self.length_path += pow(2, 1/2)

    # return range normalized in (0~1)
    def range_finder(self, _map):
        _point = self.position
        top, top_right, right, right_down = 0, 0, 0, 0
        down, left_down, left, top_left = 0, 0, 0, 0

        # find range of top
        while True:
            _point = (_point[0], _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top += 1

        # find range of top right
        _point = self.position
        while True:
            _point = (_point[0] + 1, _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top_right += 1

        # find range of right
        _point = self.position
        while True:
            _point = (_point[0] + 1, _point[1])
            if _map[_point] == -1:
                break
            else:
                right += 1

        # find range fo right down
        _point = self.position
        while True:
            _point = (_point[0] + 1, _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                right_down += 1

        # find range fo down
        _point = self.position
        while True:
            _point = (_point[0], _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                down += 1

        # find range fo left down
        _point = self.position
        while True:
            _point = (_point[0] - 1, _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                left_down += 1

        # find range of left
        _point = self.position
        while True:
            _point = (_point[0] - 1, _point[1])
            if _map[_point] == -1:
                break
            else:
                left += 1

        # find range fo top left
        _point = self.position
        while True:
            _point = (_point[0] - 1, _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top_left += 1

        _range = [top, top_right * pow(2, 1/2), right, right_down * pow(2, 1/2), down, left_down * pow(2, 1/2),
                  left, top_left * pow(2, 1/2)]
        # _range = [i * 0.01 for i in _range]

        return _range


if __name__ == '__main__':
    bot = Robot(_start_point=(10, 10))

    bot.walk(1)
    bot.walk(2)
    print(bot.start_point)
    print("bot select 1 to walk to top right")
    print("bot select 2 to walk to right")
    print("the path robot walked: ")
    print(bot.path)
    print("the length of path: " + str(bot.length_path))
    print("bot is at " + str(bot.position) + " now")
    map = np.zeros(shape=(100, 100))
    # boundary
    map[0, :] = -1
    map[99, :] = -1
    map[:, 0] = -1
    map[:, 99] = -1
    map[30:70, 30:70] = -1
    print("the range finder return: ")
    print(bot.range_finder(map))


