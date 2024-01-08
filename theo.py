import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Charger les fichiers CSV pour les données de balancement
swaying_folder_path = "DataLoggerFiles/Dataset/swaying/"
swaying_files = os.listdir(swaying_folder_path)

swaying_data = []
for file in swaying_files:
    file_path = os.path.join(swaying_folder_path, file)
    df = pd.read_csv('fusion/fusion0.csv')
    swaying_data.append(df.values)

# Charger les fichiers CSV pour les données sans balancement
nothing_folder_path = "DataLoggerFiles/Dataset/nothing/"
nothing_files = os.listdir(nothing_folder_path)

nothing_data = []
for file in nothing_files:
    file_path = os.path.join(nothing_folder_path, file)
    df = pd.read_csv(file_path)
    nothing_data.append(df.values)

swaying_data

# Créer les étiquettes pour les données (1 pour le balancement, 0 pour rien)
swaying_labels = np.ones(len(swaying_data))
nothing_labels = np.zeros(len(nothing_data))

# Concaténer les données de balancement et sans balancement
X = np.concatenate([swaying_data, nothing_data], axis=0)
y = np.concatenate([swaying_labels, nothing_labels])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assurez-vous que les dimensions de X_train et X_test sont correctes
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Définition du modèle séquentiel
model = Sequential()

# Ajout de la couche d'entrée avec le nombre d'attributs de tes données ('relu' ou 'tanh')
model.add(Dense(8, input_dim=X_train.shape[1], activation='tanh'))

#model.add(dropout)

# Ajout de la couche de sortie avec 1 neurone (correspondant à la sortie)
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle avec une fonction de perte binaire_crossentropy et un optimiseur 'adam' ou 'rmsprop'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle avec les données d'entraînement
model.fit(X_train, y_train, epochs=300, verbose=2)

# Évaluation du modèle sur les données de testa
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

if loss < 0.45 and accuracy > 0.75 and loss != 0.00 and accuracy != 1.00:
    # Générer le nom du fichier en fonction de la loss et de l'accuracy
    model_filename_h5 = f"L_{loss:.2f}_A_{accuracy:.2f}.h5"
    model_filename_keras = f"L_{loss:.2f}_A_{accuracy:.2f}.keras"

    # Enregistrer le modèle avec le nom de fichier spécifié
    model.save(model_filename_h5)
    model.save(model_filename_keras)

    print(f"Model saved as {model_filename_h5}")
else:
    print("Model not saved as it doesn't meet the criteria.")