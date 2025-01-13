

values = [         
  112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115., 126.,
  141., 135., 125., 149., 170., 170., 158., 133., 114., 140., 145., 150., 178., 163.,
  172., 178., 199., 199., 184., 162., 146., 166., 171., 180., 193., 181., 183., 218.,
  230., 242., 209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264., 272.,
  237., 211., 180., 201., 204., 188., 235., 227., 234., 264., 302., 293., 259., 229.,
  203., 229., 242., 233., 267., 269., 270., 315., 364., 347., 312., 274., 237., 278.,
  284., 277., 317., 313., 318., 374., 413., 405., 355., 306., 271., 306., 315., 301.,
  356., 348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318., 362., 348.,
  363., 435., 491., 505., 404., 359., 310., 337., 360., 342., 406., 396., 420., 472.,
  548., 559., 463., 407., 362., 405., 417., 391., 419., 461., 472., 535., 622., 606.,
  508., 461., 390., 432.,
 ]


import pandas as pd
idx = pd.date_range("1949-01-01", periods=len(values), freq="M")


passengers = pd.Series(values, index=idx, name="passengers").to_frame()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(passengers, passengers.passengers.shift(-1), shuffle=False)


import tensorflow.keras as keras
import tensorflow as tf

DROPOUT_RATIO = 0.2
HIDDEN_NEURONS = 10

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

def create_model(passengers):
  input_layer = keras.layers.Input(len(passengers.columns))

  hiden_layer = keras.layers.Dropout(DROPOUT_RATIO)(input_layer)
  hiden_layer = keras.layers.Dense(HIDDEN_NEURONS, activation='relu')(hiden_layer)

  output_layer = keras.layers.Dropout(DROPOUT_RATIO)(hiden_layer)
  output_layer = keras.layers.Dense(1)(output_layer)

  model = keras.models.Model(inputs=input_layer, outputs=output_layer)

  model.compile(loss='mse', optimizer=keras.optimizers.Adagrad(),
    metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
  return model

model = create_model(passengers)


model.fit(X_train, y_train, epochs=1000, callbacks=[callback])


predicted = model.predict(X_test)


import matplotlib.pyplot as plt

def show_result(y_test, predicted):
  plt.figure(figsize=(16, 6))
  plt.plot(y_test.index, predicted, 'o-', label="predicted")
  plt.plot(y_test.index, y_test, '.-', label="actual")

  plt.ylabel("Passengers")
  plt.legend()


show_result(y_test, predicted)


passengers["month"] = passengers.index.month.values
passengers["year"] = passengers.index.year.values

model = create_model(passengers)
X_train, X_test, y_train, y_test = train_test_split(passengers, passengers.passengers.shift(-1), shuffle=False)
model.fit(X_train, y_train, epochs=100, callbacks=[callback])
predicted = model.predict(X_test)


show_result(y_test, predicted)


from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf


DROPOUT_RATIO = 0.1
HIDDEN_NEURONS = 5


def create_model(passengers):
  scale = tf.constant(passengers.passengers.std())

  continuous_input_layer = keras.layers.Input(shape=1)

  categorical_input_layer = keras.layers.Input(shape=1)
  embedded = keras.layers.Embedding(12, 5)(categorical_input_layer)
  embedded_flattened = keras.layers.Flatten()(embedded)

  year_input = keras.layers.Input(shape=1)
  year_layer = keras.layers.Dense(1)(year_input)

  hidden_output = keras.layers.Concatenate(-1)([embedded_flattened, year_layer, continuous_input_layer])
  output_layer = keras.layers.Dense(1)(hidden_output)
  output = output_layer * scale + continuous_input_layer

  model = keras.models.Model(inputs=[
    continuous_input_layer, categorical_input_layer, year_input
  ], outputs=output)

  model.compile(loss='mse', optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
  return model


passengers = pd.Series(values, index=idx, name="passengers").to_frame()
passengers["year"] = passengers.index.year.values - passengers.index.year.values.min()
passengers["month"] = passengers.index.month.values - 1

X_train, X_test, y_train, y_test = train_test_split(passengers, passengers.passengers.shift(-1), shuffle=False)
model = create_model(X_train)
model.fit(
  (X_train["passengers"], X_train["year"], X_train["month"]),
  y_train, epochs=1000,
  callbacks=[callback]
)
predicted = model.predict((X_test["passengers"], X_test["year"], X_test["month"]))


show_result(y_test, predicted)





