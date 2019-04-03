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
from tkinter import *
from tkinter import filedialog
from PIL import Image

global model


class ModelGUI:
    global FilePath

    def __init__(self,master):


        self.master = master
        master.title("Welcome to the best image classification system!")

        self.directions = Label(master,text="Select Dataset File to open:").grid(row=0,column=0)
        FilePath = Entry(master).grid(row=0,column=1)

        self.browse_button = Button(master, text="Browse..",width=10,height=2,command=self.Browse,bg="blue").grid(row=0,column=2)
        # self.browse_button.pack()

        self.predict_button = Button(master, text="Predict", width=10,height=2,command=self.Predict,bg="blue").grid(row=2,column=2)
        # self.predict_button.pack()

        self.restart_button = Button(master, text="Restart",width=10,height=2,command=self.Restart,bg="blue").grid(row=3,column=0)
        # self.restart_button.pack()

        self.load_button = Button(master, text="Load network",width=10,height=2,command=self.NetworkLoad,bg="blue").grid(row=1,column=2)
        # self.load_button.pack()

        self.save_button = Button(master, text="Save network",width=10,height=2,command=self.Save,bg="blue").grid(row=3,column=2)
        # self.save_button.pack()

        OutPut = PanedWindow().grid(row=4)
        # OutPut.pack(fill=BOTH,expand=1)
        # test = Label(OutPut,text="testpane")
        # OutPut.add(test)



    def Browse(self):
        FileName = filedialog.askdirectory()
        FilePath.set(FileName)



    def NetworkLoad(self):
        ModelPath = filedialog.askdirectory()
        # ModelPath.set(ModelName)
        model = models.load_model(ModelPath)

    def Predict(self):
        # predictions = model.predict_generator
        # print(np.argmax(predictions[0]))
        print("Greetings!")


    def Restart(self):
        print("Greetings!")

    def Save(self):
        print("Greetings!")




class main():

    window = Tk()
    window.geometry("500x300")
    GUI = ModelGUI(window)
    window.mainloop()



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

def TrainModel():
    train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
    img_size = 128
    batch_size = 20
    t_steps = 3462/batch_size
    v_steps = 861/batch_size
    classes = 5
    flower_path = "/Users/eranedri/Documents/GitHub/Flower_Classification-DeepLearning/flowers"
    catagories = ["daisy","dandelion","rose","sunflower" ,"tulip"]
    train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
    valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')


    model = models.Sequential()

    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size,img_size,3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes,activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))

    optimizer = optimizers.Adam()
    loss = losses.categorical_crossentropy
    model.compile(loss= loss ,optimizer=optimizer ,metrics=['accuracy'])
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 1 , validation_data=valid_gen, validation_steps=v_steps)
    model.save('flowers_model.h5')
    plt_modle(model_hist)






