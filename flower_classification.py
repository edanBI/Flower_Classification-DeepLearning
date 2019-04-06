import keras
# from keras import *
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers
from keras import losses


# from tkinter import Entry
# from tkinter import Tk
# from tkinter import Button
# from tkinter import END


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

    # first load te model from the modeldir and then classified the chosen dataset from the datasetdir and pop up new window with the results of the model
    def Predict(self):
        classifier_model = models.load_model(self.ModelPath.get())

        classifier_model.compile(optimizer=optimizers.Adam(),loss=losses.sparse_categorical_crossentropy,metrics=['accuracy'])

        image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest',
                                   validation_split=0.2)
        image_data = image_generator.flow_from_directory(self.DataSetPath.get(), target_size=(128, 128), batch_size=20,
                                              class_mode='categorical', subset='validation')

        #The resulting object is an iterator that returns image_batch, label_batch pairs.
        # imagenet_labels = (["daisy", "dandelion", "rose", "sunflower", "tulip"])

        imagenet_labels = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
        imagenet_labels = np.array([key.title() for key, value in imagenet_labels])

        for image_batch, label_batch in image_data:
            result_batch = classifier_model.predict(image_batch)
            labels_batch = imagenet_labels[np.argmax(result_batch, axis=-1)]
            print(result_batch)
            break

        for n in range(len(image_data)):
            plt.imshow(image_batch[n])
            plt.title(labels_batch[n])
            plt.axis('off')
        _ = plt.suptitle("Model predictions")

        print(self.DataSetPath.get())
        print(self.ModelPath.get())

    # restart all the gui fields for new classification
    def Restart(self):
        self.DataSetPath = tk.StringVar()
        self.ModelPath = tk.StringVar()
        self.entry1.delete(0, tk.END)
        self.entry2.delete(0, tk.END)


# main program that initilized the GUI instane
class main():
    global model
    window = tk.Tk()
    window.geometry("550x200")
    GUI = ModelGUI(window)
    window.mainloop()


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
    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2,
                               width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
    img_size = 128
    batch_size = 20
    t_steps = 3462 / batch_size
    v_steps = 861 / batch_size
    classes = 5
    flower_path = "/Users/eranedri/Documents/GitHub/Flower_Classification-DeepLearning/flowers"
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
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=10, validation_data=valid_gen,

                                  validation_steps=v_steps)

    # model = Sequential()
    # model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(number_of_classes, activation='softmax'))

    # # MLP
    # def mlp_mc_model():
    #     model = Sequential()
    #
    #     model.add(Dense(128, activation='relu', input_shape=(128 * 128 * 3,)))
    #     model.add(BatchNormalization())
    #
    #     model.add(Dense(256, activation='relu'))
    #     model.add(BatchNormalization())
    #
    #     model.add(Dense(512, activation='relu'))
    #     model.add(BatchNormalization())
    #
    #     model.add(Dense(1024, activation='relu'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(10, activation='softmax'))
    #
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model



    model.save('flowers_model.h5')
    plt_modle(model_hist)
