import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
import timeit
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.random_seed import set_random_seed

seed = 0
np.random.seed(seed)
set_random_seed(seed)

# Définition des exemples entrées / sorties
donneees = pds.read_csv('fusion/fusion0.csv')

temps = donneees['T [ms]'].to_numpy()
acceleration_x = donneees['AccX [mg]'].to_numpy()
acceleration_y = donneees['AccY [mg]'].to_numpy()
acceleration_z = donneees['AccZ [mg]'].to_numpy()

# Sorties correspondantes
seuil_min = 0.35
seuil_max = 0.8
sample = 20
Y = np.ones(len(acceleration_x))
for n_exe in range(0, len(acceleration_x)):
    if (seuil_min <= acceleration_y[n_exe] <= seuil_max and
            (seuil_min <= acceleration_x[n_exe] <= seuil_max or
             seuil_min <= acceleration_z[n_exe] <= seuil_max)):
        Y[n_exe] = 0

# On étudie sur une fenêtre de "sample" valeurs
samples = [Y.iloc[i:i+sample] for i in range(0, len(Y), sample)]
Y_moy = [sample.mean() for sample in samples]

for i in range(0, len(Y_moy)):
    if(Y_moy[i] >= 0.5):
        Y_moy[i] = 1
    else:
        Y_moy[i] = 0

# Concaténation des données d'accélération
Et = np.column_stack((acceleration_x, acceleration_y, acceleration_z))
E_train, E_test, Y_train, Y_test = train_test_split(Et, Y_moy, test_size=0.33, random_state=seed)

# Créer le modèle
model = Sequential()
model.add(Dense(100, input_dim=Et.shape[1], activation='tanh'))  # Utiliser la bonne dimension d'entrée
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Entraînement du modèle
start_time = timeit.default_timer()
history = model.fit(E_train, Y_train, validation_split=0.15, epochs=300, verbose=0, batch_size=10)
print("Temps passé : %.2fs" % (timeit.default_timer() - start_time))

# plot figure
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
# Plot training & validation loss values

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# evaluate the model
scores = model.evaluate(E_test, Y_test)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

# Export du modèle
model.save('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# evaluate with the all dataset and plot
prediction = model.predict_on_batch(Et)
prediction = prediction.reshape(-1, 123)
attendues = Y.reshape(-1, 123)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(attendues, extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction attendue')
plt.xlabel('Entree 1')
plt.subplot(1, 2, 2)
plt.imshow(prediction, extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction predite')
plt.xlabel('Entree 1')
plt.ylabel('Entree 2')
plt.show()