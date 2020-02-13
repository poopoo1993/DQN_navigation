import numpy as np


def convert_string2matrix(_string):
    _string = _string.replace('\n', '')
    _string = _string.replace('][', '],[')
    _string = _string.split(',')

    matrix = list()
    for i in range(len(_string)):
        _string[i] = _string[i].replace('[', '')
        _string[i] = _string[i].replace(']', '')
        matrix.append(_string[i].split())

    for i in range(len(matrix)):
        matrix[i] = [int(j.replace('.', '')) for j in matrix[i]]

    matrix = np.array(matrix)
    return matrix


def read_map(filename):
    # open file
    file = open(filename+'.txt', 'r')
    text = file.read()
    file.close()

    # convert format
    map = convert_string2matrix(text)
    return map


if __name__ == '__main__':
    print(read_map('../map/map'))
