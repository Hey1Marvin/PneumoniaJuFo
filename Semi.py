import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

data_dir="C:\\Users\\marvi\\OneDrive\\Dokumente\\Schule\\semi_CNN_2022-23\\CNNTensorflow\\Learnset\\chest_xray\\val"
data_dir="C:\\Users\\marvi\\OneDrive\\Dokumente\\Schule\\semi_CNN_2022-23\\CNNTensorflow\\Learnset - Kopie\\chest_xray\\test"
#Data Directory aus der die Daten kommen
categories=["NORMAL", "PNEUMONIA"]
#Kategorien in die unterteilt wird
img_size= 256

#save via numpy in npz Datei
# Definieren Sie die Funktion save_images, um die Bilder und die Labels in einer .npz-Datei zu speichern
def save_images (filename, bilder, labels):
  # Speichern Sie die Arrays in einer komprimierten Datei mit np.savez
  np.savez(filename, X=bilder, labels=labels)
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden in", filename, "gespeichert")

# Definieren Sie die Funktion load_images, um die Bilder und die Labels aus einer .npz-Datei zu laden
def load_images (filename):
  # Laden Sie die Arrays aus der Datei mit np.load
  data = np.load (filename)
  # Holen Sie sich die Arrays mit den entsprechenden Schlüsseln
  bilder = data ["X"]
  labels = data ["labels"]
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden aus", filename, "geladen")
  # Geben Sie die Arrays zurück
  return bilder, labels

    
#Save via Pickle   
def savePickle(param, path = 'X.pickle'):
    pickle_out=open("X.pickle","wb")
    pickle.dump(param, pickle_out)
    pickle_out.close()
    print("Prameter saved in ", path)


def loadPickle(path = 'X.pickle'):
    pickle_in=open("X.pickle","rb")
    print("Parameter load from: ", path)
    return pickle.load(pickle_in)


# Funktion zum Plotten eines Bildes
def display_image (index, bilder, labels):
  # Überprüfen Sie, ob der Index gültig ist
  if index < 0 or index >= len (bilder):
    print ("Ungültiger Index")
    return
  # Holen Sie sich das Bild und das Label an dem gegebenen Index
  image = bilder [index]
  label = labels [index]
  # Erstellen Sie eine Abbildung, um das Bild anzuzeigen
  plt.figure ()
  plt.imshow (image)
  # Erstellen Sie einen Titel für die Abbildung, der das Label angibt
  if label == 0:
    title = "Gesund/Normal"
  elif label == 1:
    title = "Bakterielle Pneumonie"
  elif label == 2:
    title = "Virale Pneumonie"
  else:
    title = "Unbekanntes Label"
  plt.title (title)
  # Zeigen Sie die Abbildung an
  plt.show ()

#zum Einlesen der Trainingsdaten aus mehreren Verzeichnissen
def create_training_data(data_dir, categories, img_number = 4):#erschaffen und vervielfältigen Trainingdaten  
    training_data=[]
    i = 0
    for category in categories:
        print("Category: ", category)
        path=os.path.join(data_dir, category)
        class_num= categories.index(category)
        for img in tqdm(os.listdir(path)):
              
            if 'virus' in img:
                label = 1
            if 'bacteria' in img:
                label = 2
            else:
                label = 0
            try:
                img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                array_new=cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                training_data.append([array_new, label])
                #rotieren, verschieben, scheren, zoomen, flippen , fülle leere Pixel mt dem nächsten
                datagen=ImageDataGenerator(
                    rotation_range=12,
                    width_shift_range=0.12,
                    height_shift_range=0.12,
                    shear_range=0.12,
                    zoom_range=0.12,
                    horizontal_flip=True,
                    fill_mode="nearest",
                )
                img_array=img_array.reshape((1,)+img_array.shape+(1,))#damit neues array kompatibel mit IDG ist 
                i=0
                for batch in datagen.flow(img_array, batch_size=1):
                    augmented_img=batch[0].astype("uint8")#.reshape(img_size, img_size)
                    augmented_img = cv2.resize(batch[0].astype("uint8"), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                    training_data.append([augmented_img, label])
                    i+=1
                    #für den for loop
                    if i>=img_number:
                        break
                #dreimaliges vervielfältigen der Daten   
            except Exception as e:
                print("Bild fehlerhaft aufgrund folgenden Fehlers: ", e) #ersetzen eines Error mit dieser Fehlermeldung
    return training_data

#Einlesen der Trainingsdaten aus einem Vereichniss und klassifizierun über den Namen der Datei
def create_training_data2(path, img_number = 4):#erschaffen und vervielfältigen Trainingdaten  
    training_data = []
    for img in os.listdir(path):
        print("path: ", os.listdir(path))
        #get the label of the image
        if 'virus' in img:
            label = 1
        if 'bacteria' in img:
            label = 2
        else:
            label = 0
        
        try:
            img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            array_new=cv2.resize(img_array, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            training_data.append([array_new, label])
            #rotieren, verschieben, scheren, zoomen, flippen , fülle leere Pixel mt dem nächsten
            datagen=ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode="nearest",
            )
            img_array=img_array.reshape((1,)+img_array.shape+(1,))#damit neues array kompatibel mit IDG ist 
            i=0
            for batch in datagen.flow(img_array, batch_size=1):
                augmented_img=batch[0].astype("uint8")#.reshape(img_size, img_size)
                augmented_img = cv2.resize(batch[0].astype("uint8"), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                training_data.append([augmented_img, label])
                i+=1
                #für den for loop
                if i>=img_number:
                    break
        except Exception as e:
            print("Bild fehlerhaft aufgrund folgenden Fehlers: ", e) #ersetzen eines Error mit dieser Fehlermeldung
    return training_data

def create_training_data_from_list(image, labels, img_number = 4):#erschaffen und vervielfältigen Trainingdaten  
    #training_data=[]
    for img, label in zip(image, label):
        try:
            array_new=cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            print("Array-Shape: ", array_new.shape)
            #training_data.append([array_new, label])
            #rotieren, verschieben, scheren, zoomen, flippen , fülle leere Pixel mt dem nächsten
            datagen=ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest",
            )
            img_array=img_array.reshape((1,)+img_array.shape+(1,))#damit neues array kompatibel mit IDG ist 
            i=0
            for batch in datagen.flow(img_array, batch_size=1):
                augmented_img=batch[0].astype("uint8")#.reshape(img_size, img_size)
                augmented_img = cv2.resize(batch[0].astype("uint8"), (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                image.append(augmented_img)
                labels.append(label)
                i+=1
                #für den for loop
                if i>=img_number:
                    break
            #dreimaliges vervielfältigen der Daten   
        except Exception as e:
            print("Bild fehlerhaft aufgrund folgenden Fehlers: ", e) #ersetzen eines Error mit dieser Fehlermeldung
    return image, label


training_data = create_training_data(data_dir, categories, 3)

random.shuffle(training_data)
#shufflen der Daten gegen Overfitting
X=[] 
y=[] 
for feature, label in training_data:
    X.append(feature)
    y.append(label)
X=np.array(X).reshape(-1, img_size, img_size, 1)
print("Len-images: ", len(X))
#for i in range(len(X)):
#    display_image(i, X, y)


save_images("expandedDataset2.npz", X, y)
  
    
