import numpy as np
import pickle
import random
from LoadMnist import getData, data_redistribute
import Config
from MainModel import Softmax, get_accuracy, get_vars, get_learning
from Attacks import same_value_attack, sign_flipping_attacks, sample_duplicating_attack
import logging.config


logging.config.fileConfig(fname='..\\..\\Log\\loginit.ini', disable_existing_loggers=False)
logger = logging.getLogger("infoLogger")


class BRIDGEWorker(Softmax):

    def __init__(self, para, id, workerPara, config, lr):
        """
        Initialize the solver for regular workers

        :param para: model parameter, shape(10, 784)
        :param id: id of worker
        :param workerPara: the set of regular model parameters
        :param config: configuration
        :param lr: learning step
        """
        super().__init__(para, config)
        self.id = id
        self.workerPara = workerPara
        self.lr = lr

    def aggregate(self):
        """
        Aggregate the received models (consensus step)
        """
        neighbors_para = []
        for j in range(self.config['nodeSize']):
            if (self.id, j) in Config.G.edges() or self.id == j:
                neighbors_para.append(self.workerPara[j])
        number_neighbor = len(neighbors_para)
        neighbors_para = np.array(neighbors_para)

        byzantineSize = self.config['byzantineSize']
        w_dimension = np.reshape(neighbors_para, (-1, 10 * 784)).T
        first = np.sort(w_dimension)
        second = first[:, byzantineSize: number_neighbor - byzantineSize]
        third = np.sum(second, axis=1) / (number_neighbor - 2 * byzantineSize)
        aggregate_para = np.reshape(third, (10, 784))
        return aggregate_para

    def train(self, image, label):
        """
        Updata the local model

        :param image: shape(10, 784)
        :param label: scalar
        :return:
        """
        partical_gradient = self.cal_minibatch_sto_grad(image, label)
        aggregate_para = self.aggregate()

        self.para = aggregate_para - self.lr * partical_gradient


def bridge(setting, attack):
    """
    Run BRIDGE method in iid and non-iid settings under Byzantine attacks

    :param setting: 'iid' or 'noniid'
    :param attack: same-value attacks, sign-flipping attacks
                   sample-duplicating attacks(non-iid case)
    """
    print(Config.byzantine)
    print(Config.regular)

    # Load the configurations
    conf = Config.DPSGDConfig.copy()
    num_data = int(Config.mnistConfig['trainNum'] / conf['nodeSize'])

    classification_accuracy = []
    variances = []

    # Get the training data
    image_train, label_train = getData('../../MNIST/train-images.idx3-ubyte',
                                       '../../MNIST/train-labels.idx1-ubyte')

    # Rearrange the training data to simulate the non-i.i.d. case
    if setting == 'noniid':
        image_train, label_train = data_redistribute(image_train, label_train)

    # Get the testing data
    image_test, label_test = getData('../../MNIST/t10k-images.idx3-ubyte',
                                     '../../MNIST/t10k-labels.idx1-ubyte')

    # Parameter initialization
    workerPara = np.zeros((conf['nodeSize'], 10, 784))

    # Start training
    k = 0
    last_str = '-wa'
    max_iteration = conf['iterations']
    select = random.choice(Config.regular)

    logger.info("Start!")
    while k < max_iteration:
        k += 1
        count = 0
        workerPara_memory = workerPara.copy()

        lr = get_learning(conf['learningStep'], k)  # compute decreasing learning rate

        # Byzantine attacks
        if attack != None:
            workerPara_memory, last_str = attack(workerPara_memory)

        # Regular workers receive models from their neighbors
        # and update their local models
        for id in range(conf['nodeSize']):
            para = workerPara[id]
            model = BRIDGEWorker(para, id, workerPara_memory, conf, lr)
            if setting == 'iid':
                model.train(image_train[id * num_data: (id + 1) * num_data],
                            label_train[id * num_data: (id + 1) * num_data])
                workerPara[id] = model.get_para()
            elif setting == 'noniid':
                if id in Config.regular:
                    model.train(image_train[count * num_data : (count + 1) * num_data],
                                label_train[count * num_data : (count + 1) * num_data])
                    workerPara[id] = model.get_para()
                    count += 1

        # Testing
        if k % 200 == 0 or k == 1:
            acc = get_accuracy(workerPara[select], image_test, label_test)
            classification_accuracy.append(acc)
            var = get_vars(Config.regular, workerPara)
            variances.append(var)
            logger.info('the {}th iteration acc: {}, vars: {}'.format(k, acc, var))

    print(classification_accuracy)
    print(variances)

    # Save the experiment results
    output = open("../../experiment-results/bridge"+last_str+".pkl", "wb")
    pickle.dump((classification_accuracy, variances), output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    bridge(setting='noniid', attack=sample_duplicating_attack)