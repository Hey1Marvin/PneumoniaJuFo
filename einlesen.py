# Importieren Sie die benötigten Module
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from skimage import color

# Definieren Sie die Pfade zu den Ordnern mit den Bildern
#test
#normal_path = "Learnset\\chest_xray\\val\\NORMAL" 
#pneumonia_path = "Learnset\\chest_xray\\val\\PNEUMONIA" 

normal_path = "Learnset - Kopie\\chest_xray\\test\\NORMAL" # Der Pfad zum Ordner mit den Bildern der gesunden Menschen
pneumonia_path = "Learnset - Kopie\\chest_xray\\test\\PNEUMONIA" # Der Pfad zum Ordner mit den Bildern der erkrankten Menschen

# Erstellen Sie leere Listen, um die Bilder und die Labels zu speichern
bilder2 = []
labels2 = []

# Erstellen Sie eine Funktion, um die Bilder zu lesen, zu skalieren und in NumPy-Arrays umzuwandeln
def read_and_resize (image_path, label, bilder, labels):
  # Öffnen Sie das Bild mit PIL
  image = Image.open (image_path)
  # Skalieren Sie das Bild auf 256x256 Pixel
  image = image.resize ( (256, 256))
  #Bild in Graustufen umwandeln
  image = image.convert ('L')
  # Konvertieren Sie das Bild in ein NumPy-Array
  image = np.array (image)
  if image.shape !=(256, 256): print(image.shape)
  # Fügen Sie das Bild und das Label zu den Listen hinzu
  bilder.append (image)
  labels.append (label)

def createLabel():
    X = [] #Liste der Bilder
    labels = [] #Liste der Label zu den Bildern
    # Lesen Sie alle Bilder aus dem Ordner "NORMAL" und weisen Sie ihnen das Label 0 zu
    for filename in os.listdir (normal_path):
        # Erstellen Sie den vollständigen Pfad zum Bild
        image_path = os.path.join (normal_path, filename)
        # Rufen Sie die Funktion auf, um das Bild zu lesen und zu skalieren
        read_and_resize (image_path, 0, X, labels)

    # Lesen Sie alle Bilder aus dem Ordner "PNEUMONIA" und weisen Sie ihnen das Label 1 oder 2 zu, je nachdem, ob es sich um Bakterien oder Viren handelt
    for filename in os.listdir (pneumonia_path):
        # Erstellen Sie den vollständigen Pfad zum Bild
        image_path = os.path.join (pneumonia_path, filename)
        # Bestimmen Sie das Label anhand des Dateinamens
        if "bacteria" in filename:
            label = 1
        elif "virus" in filename:
            label = 2
        else:
            label = -1 # Ungültiges Label
        # Rufen Sie die Funktion auf, um das Bild zu lesen und zu skalieren
        read_and_resize (image_path, label, X, labels)
        
    return np.array(X), np.array(labels)


# Definieren Sie die Funktion save_images, um die Bilder und die Labels in einer .npz-Datei zu speichern
def save_images (filename, bilder, labels):
  # Speichern Sie die Arrays in einer komprimierten Datei mit np.savez
  np.savez (filename, bilder=bilder, labels=labels)
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden in", filename, "gespeichert")

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



bilder, labels = load_images("learnset.npz")
print("bilderShape: ", bilder.shape)



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
  
#display_image(2000)
