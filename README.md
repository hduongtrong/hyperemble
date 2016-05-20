# hyperemble - Yet another wrapper of Tensorflow
Currently support Linear Models: Linear Regression, Lasso, Ridge, Elastic Net, Mean Absolute Deviation Regression, Logistic Regression, Linear SVM

Will support: Neural Network, Sequence to Sequence, Pointer Network, Neural GPU...

## 0. Installation
Dependencies: numpy, scipy, scikit-learn, and tensorflow (v0.8). Also keras for their datasets. 
```bash
git clone https://github.com/hduongtrong/hyperemble.git
cd hyperemble
python setup.py install
```

## 1. The basic model: 2-Class classification with Logistic Regression (SGD)
```python

# Get MNIST Dataset
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Subsetting to only digit 0 and 1
X_train = X_train[y_train <= 1]
y_train = y_train[y_train <= 1]
X_test = X_test[y_test <= 1]
y_test = y_test[y_test <= 1]

# Run the model
from hyperemble.linear_model import LogisticRegression
clf = LogisticRegression(verbose=1)
clf.fit(X_train, y_train)
# Seconds|    Iter| TrnLoss|ValScore
#       0|       0|  8.0134|  0.2636
#       0|      40|  7.0665|  0.3031
#       0|      80|  4.8897|  0.3370
#       0|     120|  3.5307|  0.3567
#       0|     160|  2.5283|  0.3938
#       0|     200|  2.1069|  0.4728
#       0|     240|  1.1661|  0.5951
#       0|     280|  0.9989|  0.6985
#       0|     320|  0.5542|  0.7695
#       0|     360|  0.6086|  0.8161
#       0|     400|  0.4703|  0.8556
#       0|     440|  0.4362|  0.8769
#       0|     480|  0.5129|  0.8934
clf.score(X_test, y_test)
# 0.92860520094562649
```

