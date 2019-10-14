from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
#Tesor Reshaping
import numpy as np
x = np.array([[10.,1.],
             [2.,3.],
             [4.,5.]])
print(x.shape)
print("\n")
x = x.reshape((6,1))
print(x)
print("\n")
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)
