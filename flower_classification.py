import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from PIL import Image


class ModelGUI:
    def __init__(self, master):
        self.master = master
        master.title("Welcome to the best image classification system!")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Browse..", command=self.greet)
        self.greet_button.pack()

        self.txt = Entry(master, width=10)
        self.label.grid(1)

        # file = filedialog.askopenfilenames()
        # dir = filedialog.askdirectory()

        self.close_button = Button(master, text="Predict", command=Predict())
        self.close_button.pack()

        self.close_button = Button(master, text="New Classifcation", command=self.greet())
        self.close_button.pack()

    def greet(self):
        print("Greetings!")



class main():
    root = Tk()
    GUI = ModelGUI(root)
    root.mainloop()




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


def NetworkLoad(PathForModel):
    model = models.load_model(PathForModel)


def Predict():
    predictions =  model.predict_generator
    print(np.argmax(predictions[0]))



