import random
import networkx as nx


def gen_graph(nodeSize, byzantine):
    """
    Randomly generate a graph where the regular workers are connected.

    :param nodeSize: the number of workers
    :param byzantine: the set of Byzantine workers
    """
    while True:
        G = nx.fast_gnp_random_graph(nodeSize, 0.7)
        H = G.copy()
        for i in byzantine:
            H.remove_node(i)
        num_connected = 0
        for _ in nx.connected_components(H):
            num_connected += 1
        if num_connected == 1:
            break
    return G


optConfig = {
    'nodeSize': 30,
    'byzantineSize': 0,

    'iterations': 5000,
    'decayWeight': 0.01,

    'batchSize': 32,
}


mnistConfig = {
    'trainNum': 60000,
    'testNum': 10000,
    'dimensions': 784,
    'classes': 10,
}

DPSGDConfig = optConfig.copy()
# DPSGDConfig['learningStep'] = 0.5       # without attack
DPSGDConfig['learningStep'] = 0.18      # same-value attack
# DPSGDConfig['learningStep'] = 0.5       # sign-flipping attack
# DPSGDConfig['learningStep'] = 0.4         # Non-iid setting

ByRDiEConfig = optConfig.copy()
# ByRDiEConfig['learningStep'] = 0.18        # without attack
# ByRDiEConfig['learningStep'] = 0.18      # same-value attack
# ByRDiEConfig['learningStep'] = 0.8       # sign-flipping attack
ByRDiEConfig['learningStep'] = 0.9       # Non-iid setting

BRIDGEConfig = optConfig.copy()
# BRIDGEConfig['learningStep'] = 0.5       # without attack
# BRIDGEConfig['learningStep'] = 0.9       # same-value attack
BRIDGEConfig['learningStep'] = 0.6       # sign-flipping attack
# BRIDGEConfig['learningStep'] = 0.4       # Non-iid setting

OursConfig = optConfig.copy()
# OursConfig['learningStep'] = 0.3
# OursConfig['penaltyPara'] = 0.005         # without attack

OursConfig['learningStep'] = 0.28
OursConfig['penaltyPara'] = 0.01        # same-value attack

# OursConfig['learningStep'] = 0.5
# OursConfig['penaltyPara'] = 0.0022        # sign-flipping attack

# OursConfig['learningStep'] = 0.4
# OursConfig['penaltyPara'] = 0.02          # Non-iid setting


# randomly generate Byzantine workers
byzantine = random.sample(range(optConfig['nodeSize']), optConfig['byzantineSize'])  # 随机选取错误节点
regular = list(set(range(optConfig['nodeSize'])).difference(byzantine))  # 正常节点的集合

# generate topology graph
G = gen_graph(optConfig['nodeSize'], byzantine)

