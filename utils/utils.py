import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片

one = [128, 128, 128]
two = [128, 0, 0]
three = [192, 192, 128]
four = [255, 69, 0]
five = [128, 64, 128]
six = [60, 40, 222]
seven = [128, 128, 0]
eight = [192, 128, 128]
nine = [64, 64, 128]
ten = [64, 0, 128]
eleven = [64, 64, 0]
twelve = [0, 128, 192]
COLOR_DICT = np.array([one, two, three, four, five, six,
                       seven, eight, nine, ten, eleven, twelve])


def show_label(y, label_size):
    y = np.squeeze(y)
    size = label_size + (3, )
    res = np.empty(size, dtype=int)

    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(12):
                if y[i][j][k] == 1:
                    res[i][j] = COLOR_DICT[k]
                    break
    plt.imshow(res)


def show_predict_image(y, label_size):
    y = np.squeeze(y)
    size = label_size + (3, )
    pre = np.empty(size, dtype=int)

    for i in range(size[0]):
        for j in range(size[1]):
            pre[i][j] = COLOR_DICT[np.argmax(y[i][j])]
    plt.imshow(pre)
