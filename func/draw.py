import numpy as np
import cv2 as cv


def init_canvas(width, height, color=(255, 255, 255)):
    canvas = np.zeros((5 * height, 5 * width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


def draw(_map, _canvas):

    color = ((255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0))
    for i in range(len(_map)):
        for j in range(len(_map[0])):
            for k in range(5):  # 5 is unit of single block
                for l in range(5):
                    if _map[i][j] == 2:
                        _canvas[i * 5 + k][j * 5 + l] = (255, 0, 0)
                    _canvas[i * 5 + k][j * 5 + l] = color[int(_map[i][j])]


def draw_path(_path, _canvas):
    for i in range(len(_path)):
        for j in range(5):  # 5 is unit of single block
            for k in range(5):
                _canvas[_path[i][0] * 5 + j][_path[i][1] * 5 + k] = (255, 0, 0)

'''
def show(_map):
    canvas = init_canvas(len(_map), len(_map[0]))
    draw(_map, canvas)
    cv.imshow('map', canvas)
    cv.waitKey(0)
'''


def show(_map, _path=[]):
    canvas = init_canvas(len(_map), len(_map[0]))
    draw(_map, canvas)
    if not _path == []:
        draw_path(_path, canvas)
    cv.imshow('map', canvas)
    cv.waitKey(1)


if __name__ == '__main__':
    map = np.zeros(shape=(100, 100))
    path = [(10, 10), (10, 11), (10, 12), (11, 12)]

    # boundary
    map[0, :] = -1
    map[99, :] = -1
    map[:, 0] = -1
    map[:, 99] = -1
    map[30:70, 30:70] = -1

    show(map, path)

