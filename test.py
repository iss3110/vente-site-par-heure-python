import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données d'entraînement et de test
train_data = pd.read_csv("dataset_train.csv")
test_data = pd.read_csv("dataset_test.csv")
# Afficher les premières lignes du jeu de données d'entraînement
print(train_data.head())

# Vérifier les types de données et les statistiques descriptives
print(train_data.info())
print(train_data.describe())

# Visualiser la distribution de la variable cible "count" (nombre total de ventes)
plt.figure(figsize=(10, 6))
sns.histplot(train_data["count"], bins=30)
plt.xlabel("Nombre total de ventes")
plt.ylabel("Fréquence")
plt.title("Distribution du nombre total de ventes")
plt.show()

# Analyse des facteurs influençant la demande en rollers :
# Visualiser la demande en rollers en fonction des variables saison, jour de la semaine et heure de la journée
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
sns.boxplot(data=train_data, x="season", y="count", ax=axes[0])
axes[0].set_xticklabels(["Printemps", "Été", "Automne", "Hiver"])
axes[0].set_xlabel("Saison")
axes[0].set_ylabel("Nombre total de ventes")
axes[0].set_title("Demande en rollers par saison")

sns.boxplot(data=train_data, x="workingday", y="count", ax=axes[1])
axes[1].set_xticklabels(["Weekend/vacances", "Jour travaillé"])
axes[1].set_xlabel("Jour de la semaine")
axes[1].set_ylabel("Nombre total de ventes")
axes[1].set_title("Demande en rollers par jour de la semaine")

sns.boxplot(data=train_data, x=train_data["datetime"].apply(lambda x: x.split()[1].split(":")[0]), y="count", ax=axes[2])
axes[2].set_xlabel("Heure de la journée")
axes[2].set_ylabel("Nombre total de ventes")
axes[2].set_title("Demande en rollers par heure de la journée")

plt.tight_layout()
plt.show()

# Prédiction du nombre total de rollers vendus pour chaque heure du jeu de données de test :

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Séparer les variables indépendantes (features) et la variable cible (target) dans les données d'entraînement et de test
X_train = train_data.drop(["count", "datetime"], axis=1)
y_train = train_data["count"]
X_test = test_data.drop("datetime", axis=1)

# Créer un modèle de régression linéaire et l'entraîner sur les données d'entraînement
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
predictions = model.predict(X_test)

# Évaluer les performances en utilisant la RMSE
rmse = mean_squared_error(test_data["count"], predictions, squared=False)
print("RMSE:", rmse)

