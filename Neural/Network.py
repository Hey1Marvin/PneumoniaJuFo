 ########### used Modules ############

# numpy for linear algebra
import numpy as np

# matplotlib for plotting the loss functions and/or accuracy
import matplotlib.pyplot as plt

# confusion matrix
from sklearn.metrics import confusion_matrix

# accuracy score
from sklearn.metrics import accuracy_score

# show progress bar
from tqdm import tqdm

# Modules for Layer and Utility functions
from .Rectifier import *
from .Layer import *


##### Classes ######


# #### [Batch Normalization class]
class BatchNormalization:
    
    def __init__(self, momentum=0.9, epsilon=1e-6):
        '''
        Der Konstruktor der BatchNormalization-Klasse. Initialisiert das Momentum und Epsilon.
        
        Parameter:
        momentum: Momentum für den gleitenden Durchschnitt
        epsilon: 𝜖, Kleine Gleitkommazahl, die zur Varianz hinzugefügt wird, um eine Division durch Null zu vermeiden
        '''
        self.epsilon = epsilon
        self.momentum = momentum

    def initialize_parameters(self, d):
        '''
        Diese Funktion initialisiert die Parameter der Batch-Normalisierungsschicht.
        
        Parameter:
        d: Form der Eingabe zur BN-Schicht
        '''
        self.gamma = np.ones((d))
        self.beta = np.zeros((d))
        self.running_mean = np.zeros((d))
        self.running_var = np.zeros((d))

    def forward(self, z, mode='train'):
        '''
        Diese Funktion führt die Vorwärtspropagation durch. Sie berechnet den Mittelwert und die Varianz der Eingabe, 
        normalisiert die Eingabe und skaliert und verschiebt sie dann.
        
        Parameter:
        z: Eingabe zur BN-Schicht
        mode: Vorwärtspass, der für das Training oder den Test verwendet wird
        
        Ausgabe:
        q: Die normalisierten, skalierten und verschobenen Eingabedaten. Diese Ausgabe wird zur nächsten Schicht im Netzwerk weitergeleitet.
        '''
        if mode=='train':
            self.batch, self.d = z.shape
            self.mu = np.mean(z, axis = 0) # 𝜇
            self.var = np.var(z, axis=0) # 𝜎^2
            self.zmu = z - self.mu # z - 𝜇
            self.ivar = 1 / np.sqrt(self.var + self.epsilon) # 𝜎𝑖𝑛𝑣
            self.zhat = self.zmu * self.ivar
            q = self.gamma*self.zhat + self.beta # ql
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        elif mode=='test':
            q = (z - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            q = self.gamma*q + self.beta
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        return q

    def backpropagation(self, dq):
        '''
        Diese Funktion führt die Rückwärtspropagation durch. Sie berechnet die Gradienten der Skalierungs- und Verschiebungsparameter 
        und den Gradienten der Eingabe.
        
        Parameter:
        dq: Gradient der Ausgabe
        
        Ausgabe:
        dz: Der Gradient der Eingabe zur BN-Schicht. Dieser Gradient wird zur vorherigen Schicht im Netzwerk zurückgegeben und zur Aktualisierung der Gewichte und des Bias in dieser Schicht verwendet.
        '''
        self.dgamma = np.sum(dq * self.zhat, axis=0)
        self.dbeta = np.sum(dq, axis=0)
        dzhat = dq * self.gamma
        dvar = np.sum(dzhat * self.zmu * (-.5) * (self.ivar**3), axis=0)
        dmu = np.sum(dzhat * (-self.ivar), axis=0)
        dz = dzhat * self.ivar + dvar * (2/self.batch) * self.zmu + (1/self.m)*dmu
        return dz

    def update(self, learnrate, batch, k):
        '''
        Diese Funktion aktualisiert die Skalierungs- und Verschiebungsparameter basierend auf den während der Rückwärtspropagation 
        berechneten Gradienten.
        
        Parameter:
        learnrate: Lernrate
        batch: Batch-Größe (Anzahl der Proben im Batch)
        k: Iterationsnummer
        '''
        self.gamma -= self.dgamma*(learnrate/batch)
        self.beta -= self.dbeta*(learnrate/batch)
        
        


# #### Network
#Build an Neural Network

class Network:
    def __init__(self, layers=None):
        '''
        Erstellt ein sequentielles CNN-Modell.

        Parameters
        ----------
        layers : list, optional
            Eine Liste von Schichten, die dem Modell hinzugefügt werden sollen.
            Wenn None, wird eine leere Liste erstellt. Der Standardwert ist None.
        '''
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.network_architecture_called = False # Ein Attribut, das angibt, ob die Architektur des Modells berechnet wurde

    def add(self, layer):
        '''
        Fügt eine Schicht zum Modell hinzu.

        Parameters
        ----------
        layer : object
            Ein Objekt, das eine Schicht repräsentiert, z.B. Conv2D, Dense, etc.
        '''
        # Fügt die Schicht zur Liste der Schichten hinzu
        self.layers.append(layer)

    def Input(self, input_shape):
        '''
        Definiert die Eingabeform des Modells.

        Parameters
        ----------
        input_shape : tuple
            Ein Tupel, das die Form der Eingabedaten angibt, z.B. (3, 32, 32) für RGB-Bilder mit 32x32 Pixeln.
        '''
        self.d = input_shape # Die Dimension der Eingabe
        self.architecture = [self.d] # Eine Liste, die die Ausgabeform jeder Schicht speichert
        self.layer_name = ["Input"] # Eine Liste, die die Namen jeder Schicht speichert

    def network_architecture(self):
        '''
        Berechnet die Architektur des Modells basierend auf den hinzugefügten Schichten.
        '''
        for layer in self.layers: # Iteriert über jede Schicht in der Liste
            if isinstance(layer, Conv2D): # Wenn die Schicht eine Conv2D-Schicht ist
                if layer.input_shape_x is not None: # Wenn die Schicht eine Eingabeform definiert hat
                    self.Input(layer.input_shape_x) # Ruft die Input-Methode mit dieser Form auf
                layer.get_dimensions(self.architecture[-1]) # Berechnet die Ausgabeform der Schicht basierend auf der vorherigen Schicht
                self.architecture.append(layer.output_shape) # Fügt die Ausgabeform zur Architekturliste hinzu
                self.layer_name.append(layer.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            elif isinstance(layer, (Flatten, Pooling2D)): # Wenn die Schicht eine Flatten- oder Pooling2D-Schicht ist
                layer.get_dimensions(self.architecture[-1]) # Berechnet die Ausgabeform der Schicht basierend auf der vorherigen Schicht
                self.architecture.append(layer.output_shape) # Fügt die Ausgabeform zur Architekturliste hinzu
                self.layer_name.append(layer.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            elif isinstance(layer, Dense): # Wenn die Schicht eine Dense-Schicht ist
                self.architecture.append(layer.neurons) # Fügt die Anzahl der Neuronen zur Architekturliste hinzu
                self.layer_name.append(layer.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            else: # Wenn die Schicht eine andere Art von Schicht ist
                self.architecture.append(self.architecture[-1]) # Fügt die gleiche Ausgabeform wie die vorherige Schicht zur Architekturliste hinzu
                self.layer_name.append(layer.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu

        self.layers = list(filter(None, self.layers)) # Entfernt alle None-Elemente aus der Schichtenliste
        try:
            idx = self.layer_name.index("NoneType") # Sucht nach dem Index eines NoneType-Elements in der Namensliste
            del self.layer_name[idx] # Löscht das Element an diesem Index aus der Namensliste
            del self.architecture[idx] # Löscht das Element an diesem Index aus der Architekturliste
        except:
            pass # Wenn kein NoneType-Element gefunden wurde, tue nichts

    def summary(self):
        '''
        Zeigt eine Zusammenfassung des Modells an, einschließlich der Schichttypen, der Ausgabeformen und der Anzahl der Parameter.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern druckt die Zusammenfassung auf dem Bildschirm aus.
        '''
        if self.network_architecture_called==False: # Wenn die Architektur des Modells noch nicht berechnet wurde
            self.network_architecture() # Ruft die Methode network_architecture auf, um die Architektur zu berechnen
            self.network_architecture_called = True # Setzt das Attribut network_architecture_called auf True
        len_assigned = [45, 26, 15] # Eine Liste von Längen, die für die Spalten der Zusammenfassung zugewiesen werden
        count = {'Dense': 1, 'Activation': 1, 'Input': 1,
                'BatchNormalization': 1, 'Dropout': 1, 'Conv2D': 1,
                'Pooling2D': 1, 'Flatten': 1} # Ein Wörterbuch, das die Anzahl jeder Schichtart speichert

        col_names = ['Layer (type)', 'Output Shape', '# of Parameters'] # Eine Liste von Spaltennamen für die Zusammenfassung

        print("Model: CNN") # Druckt den Namen des Modells
        print('-'*sum(len_assigned)) # Druckt eine Trennlinie
        
        text = '' # Initialisiert einen leeren Text
        for i in range(3): # Iteriert über die drei Spalten
            text += col_names[i] + ' '*(len_assigned[i]-len(col_names[i])) # Fügt den Spaltennamen und die erforderlichen Leerzeichen zum Text hinzu
        print(text) # Druckt den Text

        print('='*sum(len_assigned)) # Druckt eine Trennlinie

        total_params = 0 # Initialisiert die Gesamtzahl der Parameter auf 0
        trainable_params = 0 # Initialisiert die Anzahl der trainierbaren Parameter auf 0
        non_trainable_params = 0 # Initialisiert die Anzahl der nicht trainierbaren Parameter auf 0

        for i in range(len(self.layer_name)): # Iteriert über jede Schicht in der Namensliste
            # layer name
            layer_name = self.layer_name[i] # Speichert den Namen der Schicht
            name = layer_name.lower() + '_' + str(count[layer_name]) + ' ' + '(' + layer_name + ')' # Erstellt einen eindeutigen Namen für die Schicht mit ihrer Nummer und ihrem Typ
            count[layer_name] += 1 # Erhöht die Anzahl dieser Schichtart um 1

            # output shape
            try: # Versucht, die Ausgabeform der Schicht als Tupel zu erstellen
                out = '(None, ' # Beginnt das Tupel mit None für die Batch-Dimension
                for n in range(len(self.architecture[i])-1): # Iteriert über die restlichen Dimensionen außer der letzten
                    out += str(self.architecture[i][n]) + ', ' # Fügt die Dimension und ein Komma zum Tupel hinzu
                out += str(self.architecture[i][-1]) + ')' # Fügt die letzte Dimension und eine schließende Klammer zum Tupel hinzu
            except: # Wenn die Ausgabeform keine Tupel ist
                out = '(None, ' + str(self.architecture[i]) + ')' # Erstellt die Ausgabeform als Tupel mit nur einer Dimension

            # number of params
            if layer_name=='Dense': # Wenn die Schicht eine Dense-Schicht ist
                h0 = self.architecture[i-1] # Speichert die Anzahl der Eingangsneuronen
                h1 = self.architecture[i] # Speichert die Anzahl der Ausgangsneuronen
                if self.layers[i-1].use_bias: # Wenn die Schicht einen Bias-Vektor verwendet
                    params = h0*h1 + h1 # Berechnet die Anzahl der Parameter als das Produkt der Neuronen plus die Anzahl der Ausgangsneuronen
                else: # Wenn die Schicht keinen Bias-Vektor verwendet
                    params = h0*h1 # Berechnet die Anzahl der Parameter als das Produkt der Neuronen
                total_params += params # Addiert die Anzahl der Parameter zur Gesamtzahl hinzu
                trainable_params += params # Addiert die Anzahl der Parameter zur Anzahl der trainierbaren Parameter hinzu
            elif layer_name=='BatchNormalization': # Wenn die Schicht eine BatchNormalization-Schicht ist
                h = self.architecture[i] # Speichert die Anzahl der Merkmale
                params = 4*h # Berechnet die Anzahl der Parameter als das Vierfache der Merkmale
                trainable_params += 2*h # Addiert die Hälfte der Parameter zur Anzahl der trainierbaren Parameter hinzu
                non_trainable_params += 2*h # Addiert die Hälfte der Parameter zur Anzahl der nicht trainierbaren Parameter hinzu
                total_params += params # Addiert die Anzahl der Parameter zur Gesamtzahl hinzu
            elif layer_name=='Conv2D': # Wenn die Schicht eine Conv2D-Schicht ist
                layer = self.layers[i-1] # Speichert die Schicht als ein Objekt
                if layer.use_bias: # Wenn die Schicht einen Bias-Vektor verwendet
                    add_b = 1 # Speichert eine zusätzliche Einheit für den Bias
                else: # Wenn die Schicht keinen Bias-Vektor verwendet
                    add_b = 0 # Speichert keine zusätzliche Einheit für den Bias
                params = ((layer.inputC * layer.kernelH * layer.kernelW) + add_b) * layer.F # Berechnet die Anzahl der Parameter als das Produkt der Eingangskanäle, der Kernelhöhe, der Kernelbreite und der Anzahl der Filter plus die zusätzliche Einheit für den Bias
                trainable_params += params # Addiert die Anzahl der Parameter zur Anzahl der trainierbaren Parameter hinzu
                total_params += params # Addiert die Anzahl der Parameter zur Gesamtzahl hinzu
            else: # Wenn die Schicht eine andere Art von Schicht ist
                # Pooling, Dropout, Flatten, Input
                params = 0 # Speichert die Anzahl der Parameter als 0
            names = [name, out, str(params)] # Erstellt eine Liste mit dem Namen, der Ausgabeform und der Anzahl der Parameter der Schicht

            # print this row
            text = '' # Initialisiert einen leeren Text
            for j in range(3): # Iteriert über die drei Spalten
                text += names[j] + ' '*(len_assigned[j]-len(names[j])) # Fügt den Namen, die Ausgabeform oder die Anzahl der Parameter und die erforderlichen Leerzeichen zum Text hinzu
            print(text) # Druckt den Text
            if i!=(len(self.layer_name)-1): # Wenn dies nicht die letzte Schicht ist
                print('-'*sum(len_assigned)) # Druckt eine Trennlinie
            else: # Wenn dies die letzte Schicht ist
                print('='*sum(len_assigned)) # Druckt eine Trennlinie

        print("Total params:", total_params) # Druckt die Gesamtzahl der Parameter
        print("Trainable params:", trainable_params) # Druckt die Anzahl der trainierbaren Parameter
        print("Non-trainable params:", non_trainable_params) # Druckt die Anzahl der nicht trainierbaren Parameter
        print('-'*sum(len_assigned)) # Druckt eine Trennlinie
    
    def compile(self, cost_type, optimizer_type):
        '''
        Kompiliert das Modell mit einer Kostenfunktion und einem Optimierer.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.
        cost_type : str
            Der Name der Kostenfunktion, die für das Modell verwendet werden soll, z.B. "cross-entropy" oder "mse".
        optimizer_type : str
            Der Name des Optimierers, der für das Modell verwendet werden soll, z.B. "sgd" oder "adam".

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern setzt die Attribute cost, cost_type und optimizer_type der Instanz.
        '''
        self.cost = Cost(cost_type) # Erstellt ein Objekt der Klasse Cost mit der angegebenen Kostenfunktion
        self.cost_type = cost_type # Speichert den Namen der Kostenfunktion als Attribut
        self.optimizer_type = optimizer_type # Speichert den Namen des Optimierers als Attribut

    def initialize_parameters(self):
        '''
        Initialisiert die Parameter des Modells basierend auf den hinzugefügten Schichten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern setzt die Parameter der Schichten als Attribute der Instanz.
        '''
        if not self.network_architecture_called: # Wenn die Architektur des Modells noch nicht berechnet wurde
            self.network_architecture() # Ruft die Methode network_architecture auf, um die Architektur zu berechnen
            self.network_architecture_called = True # Setzt das Attribut network_architecture_called auf True
        for i, layer in enumerate(self.layers): # Iteriert über jede Schicht in der Liste der Schichten
            if isinstance(layer, (Dense, Conv2D)): # Wenn die Schicht eine Dense- oder Conv2D-Schicht ist
                #print("Layer: ", layer.__class__.__name__, " input: ", self.architecture[i])
                layer.initialize_parameters(self.architecture[i], self.optimizer_type) # Ruft die Methode initialize_parameters der Schicht auf, um die Parameter zu initialisieren
            elif isinstance(layer, BatchNormalization): # Wenn die Schicht eine BatchNormalization-Schicht ist
                layer.initialize_parameters(self.architecture[i]) # Ruft die Methode initialize_parameters der Schicht auf, um die Parameter zu initialisieren


    
    def fit(self, X, y, epochs=10, batch_size=5, learnrate=1, X_val=None, y_val=None, verbose=1, learnrate_decay=None, **kwargs):
        '''
        Trainiert das Modell mit den gegebenen Trainingsdaten und optionalen Validierungsdaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.
        X : array-like
            Die Eingabedaten für das Training, z.B. ein Numpy-Array oder eine Liste von Arrays.
        y : array-like
            Die Ausgabedaten für das Training, z.B. ein Numpy-Array oder eine Liste von Arrays.
        epochs : int, optional
            Die Anzahl der Epochen, die das Modell trainieren soll. Der Standardwert ist 10.
        batch_size : int, optional
            Die Größe der Minibatches, die für das Training verwendet werden sollen. Der Standardwert ist 5.
        learnrate : float, optional
            Die Lernrate, die für den Optimierer verwendet werden soll. Der Standardwert ist 1.
        X_val : array-like, optional
            Die Eingabedaten für die Validierung, z.B. ein Numpy-Array oder eine Liste von Arrays. Wenn None, wird keine Validierung durchgeführt. Der Standardwert ist None.
        y_val : array-like, optional
            Die Ausgabedaten für die Validierung, z.B. ein Numpy-Array oder eine Liste von Arrays. Wenn None, wird keine Validierung durchgeführt. Der Standardwert ist None.
        verbose : int, optional
            Ein Schalter, der angibt, ob die Trainings- und Validierungsergebnisse nach jeder Epoche gedruckt werden sollen. Wenn 1, werden die Ergebnisse gedruckt. Wenn 0, werden die Ergebnisse nicht gedruckt. Der Standardwert ist 1.
        learnrate_decay : function, optional
            Eine Funktion, die die Lernrate nach jeder Iteration anpasst. Wenn None, wird keine Lernratenanpassung durchgeführt. Der Standardwert ist None.
        **kwargs : dict, optional
            Zusätzliche Schlüsselwortargumente, die an die Funktion learnrate_decay übergeben werden sollen.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern aktualisiert die Parameter des Modells und speichert die Trainings- und Validierungshistorie als Attribute der Instanz.
        '''
        self.history = {'Training Loss': [],'Validation Loss': [], 'Training Accuracy': [],  'Validation Accuracy': []} # Erstellt ein Wörterbuch, das die Trainings- und Validierungshistorie speichert
        iterations = 0 # Initialisiert die Anzahl der Iterationen auf 0
        self.batch = batch_size # Speichert die Größe der Minibatches als Attribut
        self.initialize_parameters() # Ruft die Methode initialize_parameters auf, um die Parameter des Modells zu initialisieren
        total_num_batches = np.ceil(len(X)/batch_size) # Berechnet die Gesamtzahl der Minibatches

        for epoch in range(epochs): # Iteriert über jede Epoche
            cost_train = 0 # Initialisiert die Trainingskosten auf 0
            num_batches = 0 # Initialisiert die Anzahl der Minibatches auf 0
            y_pred_train = [] # Initialisiert eine Liste, die die Vorhersagen des Modells für die Trainingsdaten speichert
            y_train = [] # Initialisiert eine Liste, die die tatsächlichen Ausgaben für die Trainingsdaten speichert

            print(f'\nEpoch: {epoch+1}/{epochs}') # Druckt die aktuelle Epoche

            for i in tqdm(range(0, len(X), batch_size)): # Iteriert über jede Minibatch
                X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
                y_batch = y[i:i+batch_size] # Extrahiert die Ausgabedaten für die Minibatch

                Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten

                # feed-forward
                for layer in self.layers: # Iteriert über jede Schicht im Modell
                    Z = layer.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen

                # calculating training accuracy
                if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                    y_pred_train += np.argmax(Z, axis=1).tolist() # Fügt die Vorhersagen des Modells für die Minibatch zur Liste der Vorhersagen hinzu
                    y_train += np.argmax(y_batch, axis=1).tolist() # Fügt die tatsächlichen Ausgaben für die Minibatch zur Liste der Ausgaben hinzu

                # calculating the loss
                cost_train += self.cost.get_cost(Z, y_batch) / self.batch # Berechnet die Kosten für die Minibatch und addiert sie zu den Trainingskosten

                # calculating dL/daL (last layer backprop error)
                dZ = self.cost.get_d_cost(Z, y_batch) # Berechnet den Fehler der letzten Schicht
                # backpropagation
                for layer in self.layers[::-1]: # Iteriert über jede Schicht im Modell in umgekehrter Reihenfolge
                    dZ = layer.backpropagation(dZ) # Ruft die Methode backpropagation der Schicht auf, um den Fehler an die vorherige Schicht weiterzugeben

                # Parameters update
                for layer in self.layers: # Iteriert über jede Schicht im Modell
                    if isinstance(layer, (Dense, BatchNormalization, Conv2D)): # Wenn die Schicht eine Dense-, BatchNormalization- oder Conv2D-Schicht ist
                        layer.update(learnrate, self.batch, iterations) # Ruft die Methode update der Schicht auf, um die Parameter der Schicht zu aktualisieren

                # Learning rate decay
                if learnrate_decay is not None: # Wenn eine Lernratenanpassungsfunktion angegeben ist
                    learnrate = learnrate_decay(iterations, **kwargs) # Ruft die Funktion learnrate_decay auf, um die Lernrate anzupassen

                num_batches += 1 # Erhöht die Anzahl der Minibatches um 1
                iterations += 1 # Erhöht die Anzahl der Iterationen um 1

            cost_train /= num_batches # Berechnet den Durchschnitt der Trainingskosten für die Epoche

            # printing purpose only (Training Accuracy, Validation loss and accuracy)

            text  = f'Training Loss: {round(cost_train, 4)} - ' # Erstellt einen Text, der die Trainingskosten enthält
            self.history['Training Loss'].append(cost_train) # Fügt die Trainingskosten zur Historie hinzu

            # training accuracy

            if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                accuracy_train = np.sum(np.array(y_pred_train) == np.array(y_train)) / len(y_train) # Berechnet die Trainingsgenauigkeit für die Epoche
                text += f'Training Accuracy: {round(accuracy_train, 4)}' # Fügt die Trainingsgenauigkeit zum Text hinzu
                self.history['Training Accuracy'].append(accuracy_train) # Fügt die Trainingsgenauigkeit zur Historie hinzu
            else: # Wenn die Kostenfunktion eine andere ist
                text += f'Training Accuracy: {round(cost_train, 4)}' # Fügt die Trainingskosten als Genauigkeit zum Text hinzu
                self.history['Training Accuracy'].append(cost_train) # Fügt die Trainingskosten als Genauigkeit zur Historie hinzu

            if X_val is not None: # Wenn Validierungsdaten angegeben sind
                cost_val, accuracy_val = self.evaluate(X_val, y_val, batch_size) # Ruft die Methode evaluate auf, um die Validierungskosten und -genauigkeit zu berechnen
                text += f' - Validation Loss: {round(cost_val, 4)} - ' # Fügt die Validierungskosten zum Text hinzu
                self.history['Validation Loss'].append(cost_val) # Fügt die Validierungskosten zur Historie hinzu
                text += f'Validation Accuracy: {round(accuracy_val, 4)}' # Fügt die Validierungsgenauigkeit zum Text hinzu
                self.history['Validation Accuracy'].append(accuracy_val) # Fügt die Validierungsgenauigkeit zur Historie hinzu

            if verbose:
                    print(text)
            else:
                print()
    
    def evaluate(self, X, y, batch_size=None):
        '''
        Bewertet das Modell mit den gegebenen Testdaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.
        X : array-like
            Die Eingabedaten für den Test, z.B. ein Numpy-Array oder eine Liste von Arrays.
        y : array-like
            Die Ausgabedaten für den Test, z.B. ein Numpy-Array oder eine Liste von Arrays.
        batch_size : int, optional
            Die Größe der Minibatches, die für den Test verwendet werden sollen. Wenn None, wird die Länge von X verwendet. Der Standardwert ist None.

        Returns
        -------
        cost : float
            Die Kosten des Modells für die Testdaten, berechnet mit der Kostenfunktion des Modells.
        accuracy : float
            Die Genauigkeit des Modells für die Testdaten, berechnet als der Anteil der korrekten Vorhersagen.
        '''
        if batch_size is None: # Wenn keine Batch-Größe angegeben ist
            batch_size = len(X) # Verwendet die Länge von X als Batch-Größe

        cost = 0 # Initialisiert die Kosten auf 0
        correct = 0 # Initialisiert die Anzahl der korrekten Vorhersagen auf 0
        num_batches = 0 # Initialisiert die Anzahl der Minibatches auf 0
        utility = Utility() # Erstellt ein Objekt der Klasse Utility
        Y_1hot, _ = utility.onehot(y) # Wandelt die Ausgabedaten in One-Hot-Vektoren um

        for i in range(0, len(X), batch_size): # Iteriert über jede Minibatch
            X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
            y_batch = y[i:i+batch_size] # Extrahiert die Ausgabedaten für die Minibatch
            Y_1hot_batch = Y_1hot[i:i+batch_size] # Extrahiert die One-Hot-Vektoren für die Minibatch
            Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten
            for layer in self.layers: # Iteriert über jede Schicht im Modell
                if layer.__class__.__name__=='BatchNormalization': # Wenn die Schicht eine BatchNormalization-Schicht ist
                    Z = layer.forward(Z, mode='test') # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht im Testmodus zu berechnen
                else: # Wenn die Schicht eine andere Art von Schicht ist
                    Z = layer.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen
            if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                cost += self.cost.get_cost(Z, Y_1hot_batch) / len(y_batch) # Berechnet die Kosten für die Minibatch und addiert sie zu den Gesamtkosten
                y_pred = np.argmax(Z, axis=1).tolist() # Berechnet die Vorhersagen des Modells für die Minibatch
                correct += np.sum(y_pred == y_batch) # Zählt die Anzahl der korrekten Vorhersagen für die Minibatch
            else: # Wenn die Kostenfunktion eine andere ist
                cost += self.cost.get_cost(Z, y_batch) / len(y_batch) # Berechnet die Kosten für die Minibatch und addiert sie zu den Gesamtkosten

            num_batches += 1 # Erhöht die Anzahl der Minibatches um 1

        if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
            accuracy = correct / len(y) # Berechnet die Genauigkeit des Modells für die Testdaten
            cost /= num_batches # Berechnet den Durchschnitt der Kosten für die Testdaten
            return cost, accuracy # Gibt die Kosten und die Genauigkeit zurück
        else: # Wenn die Kostenfunktion eine andere ist
            cost /= num_batches # Berechnet den Durchschnitt der Kosten für die Testdaten
            return cost, cost # Gibt die Kosten zweimal zurück

    def loss_plot(self):
        '''
        Zeigt einen Plot der Trainings- und Validierungskosten pro Epoche an.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern zeigt den Plot auf dem Bildschirm an.
        '''
        plt.plot(self.history['Training Loss'], 'k') # Plottet die Trainingskosten in schwarz
        if len(self.history['Validation Loss'])>0: # Wenn es Validierungskosten gibt
            plt.plot(self.history['Validation Loss'], 'r') # Plottet die Validierungskosten in rot
            plt.legend(['Train', 'Validation'], loc='upper right') # Fügt eine Legende mit den Namen der Kurven hinzu
            plt.title('Model Loss') # Fügt einen Titel für den Plot hinzu
        else: # Wenn es keine Validierungskosten gibt
            plt.title('Training Loss') # Fügt einen Titel für den Plot hinzu
        plt.ylabel('Loss') # Fügt eine Beschriftung für die y-Achse hinzu
        plt.xlabel('Epoch') # Fügt eine Beschriftung für die x-Achse hinzu
        plt.show() # Zeigt den Plot an

    def accuracy_plot(self):
        '''
        Zeigt einen Plot der Trainings- und Validierungsgenauigkeit pro Epoche an.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern zeigt den Plot auf dem Bildschirm an.
        '''
        plt.plot(self.history['Training Accuracy'], 'k') # Plottet die Trainingsgenauigkeit in schwarz
        if len(self.history['Validation Accuracy'])>0: # Wenn es Validierungsgenauigkeit gibt
            plt.plot(self.history['Validation Accuracy'], 'r') # Plottet die Validierungsgenauigkeit in rot
            plt.legend(['Train', 'Validation'], loc='lower right') # Fügt eine Legende mit den Namen der Kurven hinzu
            plt.title('Model Accuracy') # Fügt einen Titel für den Plot hinzu
        else: # Wenn es keine Validierungsgenauigkeit gibt
            plt.title('Training Accuracy') # Fügt einen Titel für den Plot hinzu
        plt.ylabel('Accuracy') # Fügt eine Beschriftung für die y-Achse hinzu
        plt.xlabel('Epoch') # Fügt eine Beschriftung für die x-Achse hinzu
        plt.show() # Zeigt den Plot an

    def predict(self, X, batch_size=None):
        '''
        Erzeugt Vorhersagen des Modells für die gegebenen Eingabedaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse Network.
        X : array-like
            Die Eingabedaten, für die das Modell Vorhersagen machen soll, z.B. ein Numpy-Array oder eine Liste von Arrays.
        batch_size : int, optional
            Die Größe der Minibatches, die für die Vorhersage verwendet werden sollen. Wenn None, wird die Länge von X verwendet. Der Standardwert ist None.

        Returns
        -------
        y_pred : array-like
            Die Vorhersagen des Modells für die Eingabedaten, z.B. ein Numpy-Array oder eine Liste von Arrays.
        '''
        if batch_size==None: # Wenn keine Batch-Größe angegeben ist
            batch_size = len(X) # Verwendet die Länge von X als Batch-Größe

        for i in range(0, len(X), batch_size): # Iteriert über jede Minibatch
            X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
            Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten
            for layer in self.layers: # Iteriert über jede Schicht im Modell
                if layer.__class__.__name__=='BatchNormalization': # Wenn die Schicht eine BatchNormalization-Schicht ist
                    Z = layer.forward(Z, mode='test') # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht im Testmodus zu berechnen
                else: # Wenn die Schicht eine andere Art von Schicht ist
                    Z = layer.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen
            if i==0: # Wenn dies die erste Minibatch ist
                if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                    y_pred = np.argmax(Z, axis=1).tolist() # Berechnet die Vorhersagen des Modells für die Minibatch als eine Liste von Indizes
                else: # Wenn die Kostenfunktion eine andere ist
                    y_pred = Z # Speichert die Ausgabe des Modells für die Minibatch als ein Array
            else: # Wenn dies nicht die erste Minibatch ist
                if self.cost_type=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                    y_pred += np.argmax(Z, axis=1).tolist() # Fügt die Vorhersagen des Modells für die Minibatch zur Liste der Vorhersagen hinzu
                else: # Wenn die Kostenfunktion eine andere ist
                    y_pred = np.vstack((y_pred, Z)) # Stapelt die Ausgabe des Modells für die Minibatch unter der bisherigen Ausgabe

        return np.array(y_pred) # Gibt die Vorhersagen des Modells für die Eingabedaten als ein Array zurück
    
    
