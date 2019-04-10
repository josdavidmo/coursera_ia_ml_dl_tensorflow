import os

import pandas as pd
import sqlalchemy
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

STRING_ENGINE = "mysql+pymysql://{username}:{password}@{host}:3306/{schema}"

DATABASE = {
  "username": os.environ['USERNAME'],
  "password": os.environ['PASSWORD'],
  "host": os.environ['HOST'],
  "schema": os.environ['SCHEMA']
}

AREAS_METROPOLITANAS = {
  "BARRANQUILLA": 1.0,
  "EJE CAFETERO": 2.0,
  "IBAGUE": 3.0,
  "NEIVA": 4.0,
  "BOGOTA": 5.0,
  "CARTAGENA": 6.0,
  "MEDELLIN": 7.0,
  "CALI": 8.0,
  "BUCARAMANGA": 9.0,
  "SANTA MARTA": 10.0,
  "MONTERIA": 11.0,
  "SINCELEJO": 12.0,
  "TUNJA": 13
}

CIUDADES = {
  "SOLEDAD": 1.0,
  "ARMENIA": 2.0,
  "IBAGUE": 3.0,
  "MANIZALES": 4.0,
  "PEREIRA": 5.0,
  "MALAMBO": 6.0,
  "NEIVA": 7.0,
  "BOGOTA": 8.0,
  "CARTAGENA": 9.0,
  "MEDELLIN": 10.0,
  "CALI": 11.0,
  "FLORIDABLANCA": 12.0,
  "BUCARAMANGA": 13.0,
  "BARRANQUILLA": 14.0,
  "ENVIGADO": 15.0,
  "BELLO": 16.0,
  "SANTA MARTA": 17.0,
  "DOSQUEBRADAS": 18.0,
  "JAMUNDI": 19.0,
  "GIRON": 20.0,
  "MONTERIA": 21.0,
  "CALDAS": 22.0,
  "SABANETA": 23.0,
  "PIEDECUESTA": 24.0,
  "ITAGUI": 25.0,
  "PALMIRA": 26.0,
  "PUERTO COLOMBIA": 27.0,
  "GALAPA": 28.0,
  "SINCELEJO": 29.0,
  "BARANOA": 30.0,
  "SABANALARGA": 31.0,
  "YUMBO": 32.0,
  "LA ESTRELLA": 33.0,
  "TUNJA": 34.0
}

CANALES = {
  "Tienda": 1.0,
  "Cafeteria": 2.0,
  "AutoServicio": 3.0,
  "Panaderia": 4.0,
  "Drogueria": 5.0,
  "Licoreria": 6.0,
  "Comidas Rapidas": 7.0,
  "Green Channel": 8.0,
  "LicoBares": 9.0,
  "Restaurante": 10.0,
  "Pasteleria": 11.0,
  "LicoBar": 12.0,
  "Bar": 13.0,
  "Billar": 14.0,
  "Cancha de Futbol": 15.0,
  "Tienda Marginal": 16.0,
  "Cancha de Tejo": 17.0
}

engine = sqlalchemy.create_engine(STRING_ENGINE.format(**DATABASE))


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.ylim([0, 5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.ylim([0, 20])
  plt.legend()
  plt.show()


query = "SELECT CAST(CONCAT(ec.anio, ec.ciclo) AS INT) as ciclo, " \
        "ec.area_metropolitana, " \
        "ec.ciudad, " \
        "ec.localidad, " \
        "ec.canal, " \
        "ec.canasta_id, " \
        "ec.total / em.n as DN " \
        "FROM encuesta_canasta as ec " \
        "INNER JOIN encuesta_mercado em on ec.encuesta_mercado_id = em.id"
dataset = pd.read_sql(query, engine)
dataset["area_metropolitana"] = dataset["area_metropolitana"].apply(
  lambda x: AREAS_METROPOLITANAS[x])
dataset["ciudad"] = dataset["ciudad"].apply(lambda x: CIUDADES[x])
dataset["canal"] = dataset["canal"].apply(lambda x: CANALES[x])

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


xs = norm(train_dataset).values[:, 0:6]
ys = norm(train_dataset).values[:, 6]

model = keras.Sequential([
  keras.layers.Dense(6,
                     input_shape=[6],
                     activation=tf.nn.relu),
  keras.layers.Dense(49,
                     activation=tf.nn.relu),
  keras.layers.Dense(20,
                     activation=tf.nn.relu),
  keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.RMSprop(0.001),
              loss=keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error',
                       'mean_absolute_error',
                       'accuracy'])

history = model.fit(xs,
                    ys,
                    epochs=10,
                    workers=4,
                    use_multiprocessing=True,
                    verbose=1)

plot_history(history)

