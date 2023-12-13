import matplotlib.pyplot as plt
import pandas as pds

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pds.read_csv('BalancierInfini.csv')

    # Créer un gyro
    # plt.figure(figsize=(15, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(donnees['T [ms]'], donnees['GyroX [mdps]'], label='AccX en fonction de T')
    # plt.plot(donnees['T [ms]'], donnees['GyroY [mdps]'], label='AccY en fonction de T')
    # plt.plot(donnees['T [ms]'], donnees['GyroZ [mdps]'], label='AccZ en fonction de T')

    # plt.xlabel('Temps (T) en ms')
    # plt.ylabel('Gyroscope en mdps')
    # plt.title("Graphique de la l'acceleration de Z en fonction de T")
    # plt.legend()
    #plt.grid(True)

    # Créer un accelero
    # plt.subplot(4, 1, 4)
    plt.figure(1)
    plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['AccY [mg]'], label='AccY en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['AccZ [mg]'], label='AccZ en fonction de T')

    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération en mg')
    plt.title("Graphique de la l'acceleration de Y en fonction de T")
    plt.legend()
    plt.grid(True)
    plt.show()
