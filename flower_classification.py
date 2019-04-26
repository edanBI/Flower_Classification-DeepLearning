import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import itertools
import tkinter as tk
from tkinter import filedialog
import csv

tf.logging.set_verbosity(tf.logging.ERROR)


# this part for upload exist model and check his Prediction capabilities
# system interface
class ModelGUI:

    def __init__(self, master):
        self.master = master
        self.DataSetPath = tk.StringVar()
        self.ModelPath = tk.StringVar()
        master.title("Welcome to the best image classification system!")
        self.Header = tk.Label(master, text="Let's Start!").grid(row=0, column=1)
        self.browse_button = tk.Button(master, text="Browse..", width=10, height=2, command=self.DatasetLoad).grid(
            row=1, column=2)
        self.directions = tk.Label(master, text="Select Dataset File to open:").grid(row=1, column=0)
        self.entry1 = tk.Entry(master, textvariable=self.DataSetPath)
        self.entry1.grid(row=1, column=1)
        self.browse_button = tk.Button(master, text="Browse..", width=10, height=2, command=self.ModelLoad).grid(row=2,
                                                                                                                 column=2)
        self.directions = tk.Label(master, text="Select Model to load:").grid(row=2, column=0)
        self.entry2 = tk.Entry(master, textvariable=self.ModelPath)
        self.entry2.grid(row=2, column=1)
        self.predict_button = tk.Button(master, text="Predict", width=25, height=2, command=self.Predict).grid(row=3,
                                                                                                               column=1)
        self.restart_button = tk.Button(master, text="Restart", width=25, height=2, command=self.Restart).grid(row=4,
                                                                                                               column=1)

    # user directory chooser
    def DatasetLoad(self):
        self.Dtmp = filedialog.askdirectory()
        self.DataSetPath.set(self.Dtmp)

    # function that load pre saved model
    def ModelLoad(self):
        self.file_options = {}
        self.file_options['filetypes'] = [('model files', '*.h5')]
        self.file_options['title'] = 'Model Directory:'
        self.Mtmp = filedialog.askopenfilename(**self.file_options)
        self.ModelPath.set(self.Mtmp)

    ####
    # first load te model from the modeldir and then classified the chosen dataset from the datasetdir and pop up new window with the results of the model
    def Predict(self):
        test_data = []
        DATA = self.DataSetPath.get()
        img_size = 128

        classifier_model = models.load_model(self.ModelPath.get())
        classifier_model.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        predictions = []
        flowers = []

        for file in os.listdir(DATA):
            file = os.fsdecode(file)
            if file == '.DS_Store':
                continue
            img = image.load_img(str(DATA) + "/" + file, target_size=(img_size, img_size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images = np.vstack([img])
            pred = classifier_model.predict_classes(images, batch_size=20)
            predictions.append(pred[0])
            flowers.append(file)

        print(predictions)
        print(flowers)


        for i in range(len(predictions)):
            print("{} {}\n".format(flowers[i], predictions[i]))


        with open('result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in range(len(predictions)):
                    if predictions[i] == 0:
                        writer.writerow([flowers[i], "daisy"])
                    elif predictions[i] == 1:
                        writer.writerow([flowers[i], "dandelion"])
                    elif predictions[i] == 2:
                        writer.writerow([flowers[i], "rose"])
                    elif predictions[i] == 3:
                        writer.writerow([flowers[i], "sunflower"])
                    elif predictions[i] == 4:
                        writer.writerow([flowers[i], "tulip"])


    def Restart(self):
        self.DataSetPath = tk.StringVar()
        self.ModelPath = tk.StringVar()
        self.entry1.delete(0, tk.END)
        self.entry2.delete(0, tk.END)
        del self.classifier_model


# this part for self use to train our model
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
    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
    img_size = 128
    batch_size = 20
    t_steps = 3462 / batch_size
    v_steps = 861 / batch_size
    classes = 5
    Categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    flower_path = "/Users/eranedri/Documents/GitHub/Flower_Classification-DeepLearning/flowers"

    train_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), classes=Categories, batch_size=batch_size, class_mode='categorical', subset='training', color_mode="rgb")
    valid_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), classes=Categories, batch_size=batch_size, class_mode='categorical', subset='validation', color_mode="rgb")

    # option 1
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), input_shape=(img_size, img_size, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(img_size, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    # option 2
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Conv2D(32, (3, 3)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(1))
    # model.add(layers.Activation('softmax'))

    # option 3
    # model = models.Sequential()
    # model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(img_size, img_size, 3)))
    # model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Dropout(0.3))
    # model.add(layers.MaxPooling2D(pool_size=3))
    #
    # model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Dropout(0.3))
    # model.add(layers.MaxPooling2D(pool_size=3))
    #
    # model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.Dropout(0.3))
    # model.add(layers.MaxPooling2D(pool_size=3))
    #
    # model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='elu'))
    # model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='elu'))
    # model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='elu'))
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1, activation='softmax'))

    optimizer = optimizers.Adam()
    loss = losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=10, validation_data=valid_gen, validation_steps=v_steps)
    model.save('flowers_model.h5')
    plt_modle(model_hist)


# main program that initilized the GUI instane
def main(*args):
    window = tk.Tk()
    window.geometry("550x200")
    GUI = ModelGUI(window)
    window.mainloop()
    # TrainModel()


if __name__ == '__main__':
    main()
