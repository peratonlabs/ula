# Lipschitz Certificate Demo
This demo uses layer-wise Lipschitz bounds to 
guarantee robustness against adversarial examples. 
These bounds were found to have fundamental limitations 
when using the common ReLU activation function [1], but the alternative 
activation GroupSort was shown in [2] to enable universal Lipschitz 
approximation. I.e., any 1-Lipschitz function can be approximated with 
arbitrarily deep 1-Lipschitz GroupSort network.  In [3], it was shown 
that this can also be achieved with shallow GroupSort networks.

This demo uses the Lipschitz penalization scheme described in [3].  
It is a re-implementation from the paper, however, so the numbers do not
match exactly.

## Requirements
The only requirement is PyTorch.  Installation instructions are at https://pytorch.org/get-started/locally/.  
We tested with Python3.7 and PyTorch 1.4.

## Run the Demo
The basic GroupSort demo can by run with the following command:

```
python train_mnist.py
```

The output should be something like:
```
test acc on clean examples (%): 97.370
test certified on clean examples (%): 73.730
```

To run with ReLU as a comparison, run the following command:

```
python train_mnist.py --activation relu --save-fn mnist_relu.p
```

The output should be something like:
```
test acc on clean examples (%): 93.010
test certified on clean examples (%): 10.680
```

## Refereces
1. T. Huster, J. Chiang, R. Chadha. Limitations of the Lipschitz constant as a defense against adversarial examples.  NEMESIS 2018.
2. C. Anil, J. Lucas, R. Grosse. Sorting out Lipschitz function approximation. ICML 2019.
3. J. Cohen, R. Cohen, T. Huster. Universal Lipschitz Approximation in Bounded Depth Neural Networks. arXiv:1904.04861

## Acknowledgement 
This research was partially sponsored by the U.S. Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-13-2-0045 (ARL Cyber Security CRA). The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation here on.
