import numpy as np
import math


class Softmax():
    def __init__(self, para, config):
        """
        Initialize the solver of softmax regression

        :param para: model parameter, shape(num_cats, num_features)
        :param config: configuration, type: dictionary
        """
        self.para = para
        self.config = config

    def one_hot(self, label):
        """
        Turn the label into the form of one-hot

        :param label: scalar
        """
        m = label.shape[0]
        label_onehot = [[1 if j == label[i] else 0 for j in range(10)] for i in range(m)]
        return np.array(label_onehot)

    def cal_minibatch_sto_grad(self, image, label):
        """
        Compute mini-batch gradient

        :param image: image, shape(784)
        :param label: label, scalar
        """
        select = np.random.randint(len(label))
        batchsize = self.config['batchSize']
        X = np.array(image[select: select + batchsize])
        Y = np.array(label[select: select + batchsize])
        Y = self.one_hot(Y)
        t = np.dot(self.para, X.T)
        t = t - np.max(t, axis=0)
        pro = np.exp(t) / np.sum(np.exp(t), axis=0)
        partical_gradient = - np.dot((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para
        return partical_gradient

    def cal_batch_grad(self, image, label):
        """
        Compute batch gradient

        :param image: image, shape(784)
        :param label: label, scalar
        """
        X = np.array(image)
        Y = np.array(label)
        Y = self.one_hot(Y)
        batchsize = X.shape[0]
        t = np.dot(self.para, X.T)
        t = t - np.max(t, axis=0)
        pro = np.exp(t) / np.sum(np.exp(t), axis=0)
        partical_gradient = - np.dot((Y.T - pro), X) / batchsize + self.config['decayWeight'] * self.para
        return partical_gradient

    def cal_loss(self, image, label):
        """
        Compute loss of softmax regression

        :param image: image, shape(784)
        :param label: label, scalar
        """
        X = np.array(image)
        Y = np.array(label)
        Y = self.one_hot(Y)
        num_data = X.shape[0]
        t1 = np.dot(self.para, X.T)
        t1 = t1 - np.max(t1, axis=0)
        t = np.exp(t1)
        tmp = t / np.sum(t, axis=0)
        loss = -np.sum(Y.T * np.log(tmp)) / num_data + self.config['decayWeight'] * np.sum(self.para ** 2) / 2
        return loss

    def get_para(self):
        return self.para


def predict(w, test_image, test_label):
    """
    Predict the label of the test_image

    :param w: model parameter, shape(10, 784)
    :param test_image: shape(784)
    :param test_label: scalar
    """
    mat = np.dot(w, test_image.T)
    predict_label = np.argmax(mat)
    # print("label :",test_label , "predict_label:",predict_label)
    return predict_label


def get_accuracy(w, image, label):
    """
    Compute the accuracy of the method

    :param w: model parameter, shape(10, 784)
    :param image: image, shape(784)
    :param label: label, scalar
    """
    number_sample = len(label)
    right = 0
    for i in range(number_sample):
        predict_label = predict(w, image[i], label[i])
        if predict_label == label[i]:
            right += 1
    accuracy = right / number_sample
    # print("the accuracy of training set is :", accuracy)
    return accuracy


def get_vars(regular, W):
    """
    Compute the variation of regualr model parameters

    :param regular: the set of regular workers
    :param W: the set of regular model parameters
    """

    W_regular = []
    for i in regular:
        W_regular.append(W[i])
    W_regular = np.array(W_regular)

    mean = np.mean(W_regular, axis=0)
    var = 0
    num = W_regular.shape[0]
    for i in range(num):
        var += np.linalg.norm(W_regular[i] - mean) ** 2

    return var / num


def get_learning(alpha, k):
    """
    Compute the decreasing learning step

    :param alpha: coefficient
    :param k: iteration time
    """
    return alpha / math.sqrt(k)


def get_learning_v2(alpha, k):
    return alpha / k

