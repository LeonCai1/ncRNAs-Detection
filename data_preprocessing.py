def get_data(file):
    """_generate a 2D list (matrix) from the input file_

    Args:
        file (_string_): the input file

    Returns:
        _list_: _the 2D list corresponding to the input file
                each row is the data in the input file 
                the first column is the y_label and the rest are value of 8 indexes_
    """
    matrix = []
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        l = [0]*9
        features = line.split(' ')
        for i, feature in enumerate(features):
            # we guarantee that the first one is label
            if i == 0:
                l[0] = int(feature)
            else:
                pair = feature.split(':')
                l[int(pair[0])] = float(pair[1])
        matrix.append(l)
    return matrix

def get_x_y(matrix):
    """_get x and y_label from data set_

    Args:
        matrix (_list_): _list of list_

    Returns:
        _(list, list)_: _x and y_label_
    """
    y = list(row[0] for row in matrix)
    x = list(row[1:]for row in matrix)
    return x, y


                

            
    