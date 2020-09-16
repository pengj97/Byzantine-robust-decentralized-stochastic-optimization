# Byzantine-Robust Decentralized Stochastic Optimization over Static and Time-varying Networks

Simulating code of paper https://arxiv.org/abs/2005.06276

## Environment

* python 3.7.0

## Files

* `Models`: Directory of the code of our proposed method and other benchmark methods, include DPSGD, BRIDGE, ByRDiE
* The main code can be founded in `./Models/(method)/(method).py` . You can run the code and save the experiment results in `experiment-results` directory.
  
* `Attacks.py`: Different Byzantine attacks, include same-value attacks, sign-flipping attacks, sample-duplicating attacks.

* `Config.py`: Configurations of these method. All hyper parameters like learning rate and decay weight can be tuned here

* `Draw.py`: Plot the curve of experiment results

* `LoadMnist.py`: Load MNIST dataset

* `MainModel.py`: Solver of softmax regression.

## Results
* The results of paper are stored in `paper-results` directory. 
* '-wa' : without Byzantine attacks ,  '-sv': same-value attacks
 '-sf': sign-flipping attacks,  '-noniid': attacks under non-iid data

## Dataset

* MNIST: http://yann.lecun.com/exdb/mnist/
