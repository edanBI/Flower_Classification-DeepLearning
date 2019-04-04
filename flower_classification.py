import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog


# system interface
class ModelGUI:

    def __init__(self, master):
        self.master = master
        self.DataSetPath = ""
        self.ModelPath = ""
        master.title("Welcome to the best image classification system!")
        self.Header = tk.Label(master, text="Let's Start!").grid(row=0, column=1)

        self.browse_button = tk.Button(master, text="Browse..", width=10, height=2, command=self.DatasetLoad).grid(
            row=1, column=2)
        self.directions = tk.Label(master, text="Select Dataset File to open:").grid(row=1, column=0)
        self.DataSetPath = tk.Entry(master).grid(row=1, column=1)

        self.browse_button = tk.Button(master, text="Browse..", width=10, height=2, command=self.ModelLoad).grid(row=2,
                                                                                                                 column=2)
        self.directions = tk.Label(master, text="Select Model to load:").grid(row=2, column=0)
        self.ModelPath = tk.Entry(master).grid(row=2, column=1)

        self.predict_button = tk.Button(master, text="Predict", width=25, height=2, command=self.Predict).grid(row=3,
                                                                                                               column=1)
        self.restart_button = tk.Button(master, text="Restart", width=25, height=2, command=self.Restart).grid(row=4,
                                                                                                               column=1)

    # user directory chooser
    def DatasetLoad(self):
        if self.DataSetPath is None:
            self.file_options1 = {}
            self.file_options1['filetypes'] = [('all files', '.*'),('image files!', '*.png;*.jpg')]
            self.file_options1['title'] = 'Data-set Directory:'
            self.DataSetPath = filedialog.askdirectory(**self.file_options1)

    # function that load pre saved model
    def ModelLoad(self):
        if self.ModelPath is None:
            file_options2 = {}
            file_options2['filetypes'] = [('all files', '.h5')]
            file_options2['title'] = 'Model Directory:'
            self.ModelPath = filedialog.askdirectory(**file_options2)

    # first load te model from the modeldir and then classified the chosen dataset from the datasetdir and pop up new window with the results of the model
    def Predict(self):
        model = models.load_model(self.ModelPath)
        model_res = model.predict(self.DataSetPath, batch_size=20, verbose=0, steps=861, callbacks=None)
        plt_modle(model_res)

    # print(np.argmax(predictions[0]))

    # restart all the gui fields for new classification
    def Restart(self):
        self.DataSetPath.delete(0, 'end')
        self.ModelPath.delete(0, 'end')


# main program that initilized the GUI instane
class main():
    global model
    window = tk.Tk()
    window.geometry("550x200")
    GUI = ModelGUI(window)
    window.mainloop()


# show results of the trained model
def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6));
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


# training new model on data set
def TrainModel():
    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
                               width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
    img_size = 128
    batch_size = 20
    t_steps = 3462 / batch_size
    v_steps = 861 / batch_size
    classes = 5
    flower_path = "/Users/eranedri/Documents/GitHub/Flower_Classification-DeepLearning/flowers"
    catagories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    train_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='training')
    valid_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical', subset='validation')

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    optimizer = optimizers.Adam()
    loss = losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=1, validation_data=valid_gen,
                                     validation_steps=v_steps)
    model.save('flowers_model.h5')
    plt_modle(model_hist)
