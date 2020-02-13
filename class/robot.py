import numpy as np


class robot:
    start_point = (0, 0)
    point = (0, 0)
    path = list()

    def __init__(self, _start_point=(0, 0), _path=list()):
        self.start_point = _start_point
        self.point = _start_point
        self.path = _path
        self.path.append(self.point)

    def reset(self):
        self.point = self.start_point
        self.path = list()
        self.path.append(self.point)

    def walk(self, direction):
        if direction == 0:    # walk to top
            self.point = (self.point[0], self.point[1] + -1)

        elif direction == 1:  # walk to right
            self.point = (self.point[0] + 1, self.point[1])

        elif direction == 2:  # walk to down
            self.point = (self.point[0], self.point[1] + 1)

        else:                 # walk to left
            self.point = (self.point[0] + -1, self.point[1])

    def range_finder(self, _map):
        _point = self.point
        top_range, right_range, down_range, left_range = 0, 0, 0, 0

        # find range of top
        while True:
            _point = (_point[0], _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top_range += 1
        # find range of right
        _point = self.point
        while True:
            _point = (_point[0] + 1, _point[1])
            if _map[_point] == -1:
                break
            else:
                right_range += 1
        # find range fo down
        _point = self.point
        while True:
            _point = (_point[0], _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                down_range += 1
        # find range of left
        _point = self.point
        while True:
            _point = (_point[0] + -1, _point[1])
            if _map[_point] == -1:
                break
            else:
                left_range += 1

        _range = [top_range, right_range, down_range, left_range]
        return _range


if __name__ == '__main__':
    bot = robot(_start_point=(10, 10))
    bot.walk(1)
    bot.walk(2)
    print(bot.start_point)
    print("bot.walk(1)")
    print("bot.walk(2)")
    print(bot.point)
    map = np.zeros(shape=(100, 100))
    # boundary
    map[0, :] = -1
    map[99, :] = -1
    map[:, 0] = -1
    map[:, 99] = -1
    map[30:70, 30:70] = -1

    print(bot.range_finder(map))


