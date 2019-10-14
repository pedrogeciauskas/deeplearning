#Reproduza o material complementar enviado “[3.1] Anatomy of a neural network
from keras import layers
layer = layers.Dense(32, input_shape=(784,))

from keras import models
model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))
model.summary()

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs = input_tensor, outputs=output_tensor)

from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
 loss='mse',
 metrics=['accuracy'])

#Code example
#model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)