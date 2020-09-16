import numpy as np
import struct


def readfile(path_image, path_label):
    with open(path_image, 'rb') as f1:
        buf1 = f1.read()
    with open(path_label, 'rb') as f2:
        buf2 = f2.read()
    return buf1, buf2


def get_image(buf1, number):
    """
    Read the images from the binary data

    :param buf1: the binary data of images
    :param number: the number of images
    """
    image_index = 0
    image_index += struct.calcsize('>IIII')
    im = []
    for i in range(number):
        temp = struct.unpack_from('>784B', buf1, image_index)
        temp = np.array(temp)
        im.append(temp / 255)
        image_index += struct.calcsize('>784B')
    return im


def get_label(buf2, number):
    """
    Read the labels from the binary data

    :param buf2: the binary data of labels
    :param number: the number of labels
    """
    label_index = 0
    label_index += struct.calcsize('>II')
    label = struct.unpack_from('>%dB' % number, buf2, label_index)
    return label


def get_number(buf1):
    num = struct.unpack_from('>4B',buf1, struct.calcsize('>I'))
    number = num[2] * 256 + num[3]
    return number


def getData(imagePath, labelPath):
    """
    Get the images and labels from mnist dataset

    :param imagePath:
    :param labelPath:
    """
    image_data, label_data = readfile(imagePath, labelPath)
    number = get_number(label_data)
    image = get_image(image_data, number)
    label = get_label(label_data, number)
    return image, label


def data_redistribute(image, label):
    """
    Rearrange the samples in the order of label

    :param image: image, shape(10, 784)
    :param label: label, scalar
    """
    number_sample = len(label)
    im = [[] for _ in range(10)]
    la = [[] for _ in range(10)]
    for i in range(number_sample):
        im[label[i]].append(image[i])
        la[label[i]].append(label[i])
    data_image = []
    data_label = []
    for i in range(10):
        for j in range(len(la[i])):
            data_image.append(im[i][j])
            data_label.append(la[i][j])

    return data_image, data_label


