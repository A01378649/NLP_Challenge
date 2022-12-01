import keras

#As previous callbacks can only be executed per epoch, we will define another metrics class that will store batch history
#Keras only makes accesible accuracy and loss via the logs object
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))