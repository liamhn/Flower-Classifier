import keras
from numpy import *
from pylab import *

import keras.models as km, keras.layers as kl
import keras.utils as ku

#Options:
nNodes_conv1 = 25
nNodes_conv2 = 25
numFlat = 200
d_rate = 0.7

xdata = load("50x50flowers.images.npy")
ydata = load("50x50flowers.targets.npy")

xnormed = xdata/255
ynormed = ku.to_categorical(ydata,18)

xtrain= xnormed[:1000]
ytrain = ynormed[:1000]

#can we make more data, add 2 data for each data, flipping along the two available axes
x_tend = zeros((xtrain.shape[0]*3,xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]))
y_tend = zeros((ytrain.shape[0]*3,ytrain.shape[1]))

for i in range(len(xtrain)):
    x_tend[i] = xtrain[i]
    x_tend[2*i]=flip(xtrain[i],axis =0)
    x_tend[3*i]=flip(xtrain[i],axis =1)
    
    y_tend[i] = ytrain[i]
    y_tend[2*i]=ytrain[i]
    y_tend[3*i]=ytrain[i]


xtest = xnormed[1000:]
ytest = ynormed[1000:]

model = km.Sequential()
model.add(kl.Conv2D(nNodes_conv1,kernel_size = (5,5),input_shape = xtrain[0].shape,activation = "relu"))
model.add(kl.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
model.add(kl.Conv2D(nNodes_conv2,kernel_size = (3,3),activation = 'relu'))
model.add(kl.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
model.add(kl.Flatten())
model.add(kl.Dense(numFlat,activation = "tanh"))
model.add(kl.Dropout(d_rate,name = 'dropout'))
model.add(kl.Dense(18,activation = 'softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
fit = model.fit(x_tend,y_tend,epochs = 300, batch_size = 100, verbose = 2)

score = model.evaluate(xtest, ytest)
print(score)












