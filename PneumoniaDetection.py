from Neural import Network
from Neural import Layer
from Neural import Utility
import numpy as np
import cv2



# Definieren Sie die Funktion load_images, um die Bilder und die Labels aus einer .npz-Datei zu laden
def load_images (filename):
  # Laden Sie die Arrays aus der Datei mit np.load
  data = np.load (filename)
  # Holen Sie sich die Arrays mit den entsprechenden Schlüsseln
  bilder = data ["bilder"]
  labels = data ["labels"]
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden aus", filename, "geladen")
  # Geben Sie die Arrays zurück
  return bilder, labels

#Hier muss der Absolute Pfad der Trainingsdaten als 'npz' Datei Angegeben werden (Numpy Dateityp)
X, labels = load_images('learnset.npz')

test = 500

X_train = X[test:]
X_test = X[:test]
y_train = labels[test:]
y_test = labels[:test]

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
#X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

X_train = X_train/255
X_test = X_test/255

utility = Utility()

# train validation split
X_train_new, X_val, y_train_new, y_val = utility.train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

Y_1hot_train, _ = utility.onehot(y_train_new)

input_shape = X_train_new.shape[1:]
output_dim = Y_1hot_train.shape[1]

input_shape, output_dim


print("Input: ", input_shape)
#Eingabegröße
#input_shape = (1, 254, 254) #256x256 Pixel Bild mit 1 Farbchannel (Grayscale)


net = Network()

#Erstelle das Netz

#Eingabe Schicht:
net.add(net.Input(input_shape=input_shape))

#CNN/Pooling Schichten
net.add(Layer.Conv2D(32, 3, activation_type='relu'))

net.add(Layer.Pooling2D((2,2), (2,2), pool_type='max'))

net.add(Layer.Conv2D(32, 3, activation_type='relu', padding='same'))

net.add(Layer.Pooling2D((2,2), (1,1), pool_type='max'))

#Dens Layer
net.add(Layer.Flatten())

net.add(Layer.Dense(64, activation_type='relu'))

net.add(Layer.Dense(3, activation_type = 'softmax'))


#compiliere das Netz
net.summary("Pneumonia - CNN")


batch_size = 8
epochs = 10
lr = 0.05

net.compile(cost_type="cross-entropy", optimizer_type="adam")


#lerne das Netz an
net.fit(X_train_new, Y_1hot_train, epochs, batch_size, lr, X_val, y_val)

net.save("pneumoniaParameter.pkl")

net.loss_plot
net.accuracy_plot
