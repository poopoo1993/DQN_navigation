import numpy as np


class Robot:

    def __init__(self, _start_point=(0, 0), _path=list()):
        self.start_point = _start_point
        self.point = _start_point
        self.path = _path
        self.length_path = 0
        self.path.append(self.point)

    def reset(self):
        self.point = self.start_point
        self.path = list()
        self.path.append(self.point)
        self.length_path = 0

    def walk(self, direction):
        if direction == 0:    # walk to top
            self.point = (self.point[0], self.point[1] - 1)
            self.path.append(tuple(self.point))
            self.length_path += 1

        elif direction == 1:  # walk to top_right
            self.point = (self.point[0] + 1, self.point[1] - 1)
            self.path.append(tuple(self.point))
            self.length_path += pow(2, 1/2)

        elif direction == 2:  # walk to right
            self.point = (self.point[0] + 1, self.point[1])
            self.path.append(tuple(self.point))
            self.length_path += 1

        elif direction == 3:  # walk to right_down
            self.point = (self.point[0] + 1, self.point[1] + 1)
            self.path.append(tuple(self.point))
            self.length_path += pow(2, 1/2)

        elif direction == 4:  # walk to down
            self.point = (self.point[0], self.point[1] + 1)
            self.path.append(tuple(self.point))
            self.length_path += 1

        elif direction == 5:  # walk to left_down
            self.point = (self.point[0] - 1, self.point[1] + 1)
            self.path.append(tuple(self.point))
            self.length_path += pow(2, 1/2)

        elif direction == 6:  # walk to left
            self.point = (self.point[0] - 1, self.point[1])
            self.path.append(tuple(self.point))
            self.length_path += 1

        elif direction == 7:  # walk to top_left
            self.point = (self.point[0] - 1, self.point[1] - 1)
            self.path.append(tuple(self.point))
            self.length_path += pow(2, 1/2)

    def range_finder(self, _map):
        _point = self.point
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
        _point = self.point
        while True:
            _point = (_point[0] + 1, _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top_right += 1

        # find range of right
        _point = self.point
        while True:
            _point = (_point[0] + 1, _point[1])
            if _map[_point] == -1:
                break
            else:
                right += 1

        # find range fo right down
        _point = self.point
        while True:
            _point = (_point[0] + 1, _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                right_down += 1

        # find range fo down
        _point = self.point
        while True:
            _point = (_point[0], _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                down += 1

        # find range fo left down
        _point = self.point
        while True:
            _point = (_point[0] - 1, _point[1] + 1)
            if _map[_point] == -1:
                break
            else:
                left_down += 1

        # find range of left
        _point = self.point
        while True:
            _point = (_point[0] - 1, _point[1])
            if _map[_point] == -1:
                break
            else:
                left += 1

        # find range fo top left
        _point = self.point
        while True:
            _point = (_point[0] - 1, _point[1] - 1)
            if _map[_point] == -1:
                break
            else:
                top_left += 1

        _range = [top, top_right * pow(2, 1/2), right, right_down * pow(2, 1/2), down, left_down * pow(2, 1/2),
                  left, top_left * pow(2, 1/2)]
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
    print("bot is at " + str(bot.point) + " now")
    map = np.zeros(shape=(100, 100))
    # boundary
    map[0, :] = -1
    map[99, :] = -1
    map[:, 0] = -1
    map[:, 99] = -1
    map[30:70, 30:70] = -1
    print("the range finder return: ")
    print(bot.range_finder(map))


