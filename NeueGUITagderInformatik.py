import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap import Style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import Canvas, PhotoImage, Toplevel
from PIL import Image, ImageTk


root = ttk.Window(themename="cyborg")
root.title("Solarenergie Prognose")
root.geometry("1920x1080")
root.position_center()
dateVal = None
def open(str):
 pil_image = Image.open('rnn.png')

 tk_image = ImageTk.PhotoImage(pil_image)
 label = tk.Label(root, image=tk_image)
 label.pack()
 root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
def close_event(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)




def see_date():
    global dateVal
    dateVal = cal.entry.get()
    #date_label.config(text=date)
    return dateVal

image_path = 'path'  




    
def startPredict():

     
    train( dateVal = cal.entry.get())
    display_image = PhotoImage(file='rnn.png').zoom(x=2,y=2)
    



cal = ttk.DateEntry(root, dateformat=('%Y-%m-%dT%H:%M:%S'), bootstyle="info")
cal.pack(padx=100, pady=100)


btn = ttk.Button(root, text="Datum festlegen", bootstyle="info", command=see_date)
btn.pack(padx=100, pady=100)

btn = ttk.Button(root, text="Start", bootstyle="info", command=startPredict)
btn.pack(padx=100, pady=100)



root.state('zoomed')



   
files_elect = [     "electricity_generation_in_Germany_in_2015.csv",
    "electricity_generation_in_Germany_in_2016.csv",
    "electricity_generation_in_Germany_in_2017.csv",
    "electricity_generation_in_Germany_in_2018.csv",
    "electricity_generation_in_Germany_in_2019.csv",
    "electricity_generation_in_Germany_in_2020.csv",
    "electricity_generation_in_Germany_in_2021.csv",
    "electricity_generation_in_Germany_in_2022.csv"    ]
files_elect2 = [     "electricity_generation_in_Germany_in_2015.csv",
    "electricity_generation_in_Germany_in_2016.csv",
    "electricity_generation_in_Germany_in_2017.csv",
    "electricity_generation_in_Germany_in_2018.csv",
    "electricity_generation_in_Germany_in_2019.csv",
    "electricity_generation_in_Germany_in_2020.csv",
    "electricity_generation_in_Germany_in_2021.csv",
    "electricity_generation_in_Germany_in_2022.csv",
    "energy-charts_Total_net_electricity_generation_in_Germany_in_2023.csv"
               
               ]

import pandas as pd


# Definiert eine Funktion, um eine Liste von csv Dateien für eine einzelne Variable zu verarbeiten
def process_df(files, variable_name):
  
    variable_df = pd.DataFrame()
    
   
    for file in files:
        # Liest die aktuelle CSV-Datei in einen DataFrame.
        df = pd.read_csv(file)
        # Konvertieren der Spalte 'Date (GMT+1)' in ein einheitliches Zeitformat ,rundung auf die nächste Stunde.
        df['Date (GMT+1)'] = pd.to_datetime(df['Date (GMT+1)']).dt.round('H')
        # Gruppiert nach 'Date (GMT+1)', um Duplikate zu entfernen und  den Mittelwert der Werte zu erechnen. 
        # Der Durchschnitt dieser Messungen  wird mit mean() berechnet, um einen einzelnen Wert pro Stunde zu erhalten.
        df = df.groupby('Date (GMT+1)').mean()
        # Wenn mehrere Spalten vorhanden sind, wählt die zweite Spalte (angenommen, es ist die Datenspalte).
        if len(df.columns) > 1:
            df = df.iloc[:, [1]]
        # Benennt die Spalte um in den gegebenen Variablennamen.
        df.columns = [variable_name]
        # Verkettet den aktuellen DataFrame mit dem Variable DataFrame am Index.
        variable_df = pd.concat([variable_df, df], axis=1)
    
    # Nach dem Zusammenführen aller Dateien berechnet den Mittelwert über die Spalten, um sie zu konkatenieren.
    variable_df = variable_df.mean(axis=1)
    # Gibt die zusammengefassten Daten für die Variable als DataFrame zurück.
    return variable_df.to_frame(name=variable_name)


electricity_df = process_df(files_elect, "Electricity_generation")


electricity_df2 = process_df(files_elect2, "Electricity_generation")
#wind_df = process_df([f for f in files if "Wind_Speed" in f], "Wind_Speed")

# Kombiniert die verarbeiteten DataFrames für jede Variable zu einem einzigen vollständigen DataFrame.
test_data = pd.concat([electricity_df2], axis=1)
complete_data = pd.concat([electricity_df], axis=1)

complete_data['Hour'] = complete_data.index.hour
complete_data['Month'] = complete_data.index.month

test_data['Hour'] = test_data.index.hour
test_data['Month'] = test_data.index.month


def add_season_column(df):
    # Erstellt eine neue Spalte 'Season' und initialisiere es mit 'Winter'
    # Es schließt den Monat Dezember des letzten Jahres mit ein.
    df['Jahreszeit'] = 'Winter'
    
    # Hier Jahreszeiten definieren
    df.loc[df.index.month.isin([3, 4, 5]), 'Jahreszeit'] = 'Frühling'
    df.loc[df.index.month.isin([6, 7, 8]), 'Jahreszeit'] = 'Sommer'
    df.loc[df.index.month.isin([9, 10, 11]), 'Jahreszeit'] = 'Herbst'
    
    return df

jahreszeit = add_season_column(complete_data.copy())
jahreszeit2 = add_season_column(test_data.copy())
# Füge die Daten in der Spalte 'Season' hinzu.

# Zeigt den endgültigen kombinierten DataFrame mit allen Variablen an.
complete_data = jahreszeit
test_data = jahreszeit2


season_to_numeric = {
    'Winter': 0,
    'Frühling': 1,
    'Sommer': 2,
    'Herbst': 3
 }

# Ersetze die 'Jahreszeit' Spalte durch numerische Werte.
complete_data['Jahreszeit'] = complete_data['Jahreszeit'].replace(season_to_numeric)
test_data['Jahreszeit'] = test_data['Jahreszeit'].replace(season_to_numeric)

# Show the DataFrame to confirm the changes
#complete_data

new_Data = complete_data.head(24)

test_data


if 'Global_solar_radiation' in complete_data.columns:
    complete_data = complete_data.drop('Global_solar_radiation', axis=1)
else:
    print("Column does not exist in DataFrame")

if 'Wind_Speed' in complete_data.columns:
    complete_data = complete_data.drop('Wind_Speed', axis=1)
else:
    print("Column does not exist in DataFrame")

# complete_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bestimmung der Länge des kompletten Datensatzes
n = len(complete_data)

# Aufteilung des Datensatzes in Training, Validierung und Test
# 70% der Daten für das Training, 20% für die Validierung, 10% für den Test
trainDF = complete_data[0:int(n*0.7)]
valDF = complete_data[int(n*0.7):int(n*0.9)]
testDF = complete_data[int(n*0.9):]

# Berechnung des Mittelwerts und der Standardabweichung des Trainingsdatensatzes
train_mean = trainDF.mean()
train_std = trainDF.std()

# Anzahl der Features (Spalten) im Datensatz
featuresAnzahl = complete_data.shape[1]

# Normalisierung der Trainings- Validierungs- und Testdaten
# Durch Subtraktion des Mittelwerts und Division durch die Standardabweichung
trainDF = (trainDF - train_mean) / train_std
valDF = (valDF - train_mean) / train_std
testDF = (testDF - train_mean) / train_std

# Normalisierung des gesamten Datensatzes für die Visualisierung
df_std = (complete_data - train_mean) / train_std

df_std = df_std.melt(var_name='Column', value_name='Normalized')

# Erstellung einer Violinplot-Visualisierung
 #plt.figure(figsize=(12, 6))
 #violin = sns.violinplot(x='Column', y='Normalized', data=df_std)
 #_ = violin.set_xticklabels(complete_data.keys(), rotation=90)
import numpy as np

# Definition der Klasse WindowGenerator zur Erstellung von Zeitfenstern für Zeitreihendaten.
class WindowGenerator():
  def __init__(self, inputBreite, labelBreite, shift,
              trainDF = trainDF, valDF = valDF, testDF = testDF, LabelSpalten=None):
    # Speichert die Datensätze für Training, Validierung und Test.
    self.trainDF = trainDF
    self.valDF = valDF
    self.testDF = testDF

    # Initialisiert und speichert die Namen der Label-Spalten, falls vorhanden.
    self.LabelSpalten = LabelSpalten
    if LabelSpalten is not None:
      self.LabelSpalten_indices = {name: i for i, name in
                                    enumerate(LabelSpalten)}
    # Erzeugt ein Wörterbuch der Spaltenindizes für den schnellen Zugriff.
    self.column_indices = {name: i for i, name in
                          enumerate(trainDF.columns)}

    # Berechnet Parameter für die Fenstergröße und -verschiebung.
    self.inputBreite = inputBreite
    self.labelBreite = labelBreite
    self.shift = shift

    # Berechnet die Gesamtgröße des Fensters.
    self.total_window_size = inputBreite + shift

    # Definiert die Eingabe- und Label-Indizes innerhalb des Fensters.
    self.inputSlice = slice(0, inputBreite)
    self.inputIndizes = np.arange(self.total_window_size)[self.inputSlice]

    self.labelStart = self.total_window_size - self.labelBreite
    self.labelSlice = slice(self.labelStart, None)
    self.labelIndizes = np.arange(self.total_window_size)[self.labelSlice]

  # Gibt eine repräsentative Zeichenkette der Fensterparameter zurück.
def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.inputIndizes}',
        f'Label indices: {self.labelIndizes}',
        f'Label column name(s): {self.LabelSpalten}'])

# Erzeugt ein Beispiel für ein WindowGenerator-Objekt mit bestimmten Parametern.
w1 = WindowGenerator(
    inputBreite=24,
    labelBreite=1,
    shift=24,
    trainDF=trainDF,
    valDF=valDF,
    testDF=testDF,
    LabelSpalten=['Electricity_generation']
)

# Methode zur Aufteilung des Fensters in Eingabe- und Label-Daten.
def split_window(self, features):
  # Trennt die Eingabedaten von den Label-Daten.
  inputs = features[:, self.inputSlice, :]
  labels = features[:, self.labelSlice, :]
  if self.LabelSpalten is not None:
    # Kombiniert die Label-Daten für die spezifizierten Spalten.
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.LabelSpalten],
        axis=-1)

  # Setzt die Form der Eingabe- und Label-Daten manuell, um Klarheit über ihre Dimensionen zu schaffen.
  inputs.set_shape([None, self.inputBreite, None])
  labels.set_shape([None, self.labelBreite, None])

  return inputs, labels

# Fügt die Methode split_window der Klasse WindowGenerator hinzu.
WindowGenerator.split_window = split_window
import tensorflow as tf

# Erzeugt Beispieldaten durch Stapeln von Teilen des Trainingsdatensatzes.
example_window = tf.stack([np.array(trainDF[:w1.total_window_size]),
                          np.array(trainDF[100:100+w1.total_window_size]),
                          np.array(trainDF[200:200+w1.total_window_size])])

# Wendet die split_window-Methode auf die Beispieldaten an, um Eingabe- und Label-Daten zu erhalten.
inputsBeispiel, labelsBeispiel = w1.split_window(example_window)

# Druckt die Formen der Beispieldaten, Eingabe- und Label-Daten.
 #print('Die Formen sind: (batch, time, features)')
 #print(f'Fensterform: {example_window.shape}')
 #print(f'Inputsform: {inputsBeispiel.shape}')
 #print(f'Labelsform: {labelsBeispiel.shape}')
 # Fügt der Instanz von WindowGenerator ein Attribut 'example' hinzu, das ein Beispiel von Eingabe- und Label-Daten enthält.
w1.example = inputsBeispiel, labelsBeispiel

# Definiert eine Plot-Methode für die WindowGenerator-Klasse.
def plot(self, model=None, plotSpalte='Electricity_generation', maxSubplots=3):
  # Extrahiert die Eingabe- und Label-Daten aus dem 'example'-Attribut der Klasse.
 inputs, labels = self.example

  # Erstellt ein Plot-Fenster mit definierter Größe.
 plt.figure(figsize=(12, 8))

  # Ermittelt den Index der Spalte, die geplottet werden soll.
 plot_col_index = self.column_indices[plotSpalte]

  # Begrenzt die Anzahl der Subplots auf das Minimum von maxSubplots und der Anzahl der Beispiele.
 max_n = min(maxSubplots, len(inputs))

  # Erstellt für jedes Beispiel einen Subplot.
 for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plotSpalte} [normalisiert]')

    # Plottet die Eingabedaten für die ausgewählte Spalte.
    plt.plot(self.inputIndizes, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=-10)

    # Ermittelt den Index der Label-Spalte.
    if self.LabelSpalten:
      label_col_index = self.LabelSpalten_indices.get(plotSpalte, None)
    else:
      label_col_index = plot_col_index

    # Überspringt den Rest der Schleife, wenn kein Label-Index vorhanden ist.
    if label_col_index is None:
      continue

    # Plottet die Label-Daten.
    plt.scatter(self.labelIndizes, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)

    # Plottet die Vorhersagen des Modells, falls ein Modell angegeben ist.
    if model is not None:
      prognose = model(inputs)
      plt.scatter(self.labelIndizes, prognose[n, :, label_col_index],
                  marker='X', edgecolors='k', label='prognose',
                  c='#ff7f0e', s=64)

    # Fügt im ersten Subplot eine Legende hinzu.
    if n == 0:
      plt.legend()

  # Fügt eine x-Achsen-Beschriftung hinzu.
    plt.xlabel('Zeit [std]')

# Fügt die plot-Methode zur WindowGenerator-Klasse hinzu.
 WindowGenerator.plot = plot

 # Definiert eine Methode zur Erstellung eines TensorFlow-Datensatzes aus einem gegebenen DataFrame.
def make_dataset(self, data):
  # Konvertiert die Daten in ein NumPy-Array mit dem Datentyp float32.
   data = np.array(data, dtype=np.float32)
  # Erstellt einen Zeitreihen-Datensatz aus dem Array, mit spezifizierten Parametern.
   ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  # Wendet die split_window-Methode auf den Datensatz an, um ihn in Eingabe- und Ausgabe-Daten aufzuteilen.
   ds = ds.map(self.split_window)

   return ds

# Fügt die make_dataset-Methode zur WindowGenerator-Klasse hinzu.
WindowGenerator.make_dataset = make_dataset

# Definiert Eigenschaften, um Trainings-, Validierungs- und Testdatensätze als TensorFlow-Datensätze bereitzustellen.
@property
def train(self):
  return self.make_dataset(self.trainDF)

@property
def val(self):
  return self.make_dataset(self.valDF)

@property
def test(self):
  return self.make_dataset(self.testDF)

# Definiert eine Eigenschaft, um ein Beispielbatch für die Visualisierungszwecke zu liefern.
@property
def example(self):
  # Prüft, ob ein Beispielbatch bereits gespeichert ist.
  result = getattr(self, 'beispiel', None)
  if result is None:
    # Holt ein Beispielbatch aus dem Trainingsdatensatz.
    result = next(iter(self.train))
    # Speichert es für zukünftige Verwendungen.
    self.beispiel = result
  return result

# Fügt die Eigenschaftsmethoden zur WindowGenerator-Klasse hinzu.
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Überprüft die Spezifikation des Elementes des Trainingsdatensatzes.
w1.train.element_spec
 # Definiert eine Konstante für die maximale Anzahl von Trainingsepochen.
MAX_EPOCHS = 150

# Definiert eine Funktion zum Kompilieren und Trainieren des Modells.
def kompilieren(model, window, patience=2):
  # Erstellt eine EarlyStopping-Rückruffunktion, die das Training frühzeitig beendet,
  # wenn der Validierungsverlust für eine bestimmte Anzahl von Epochen ('patience') nicht abnimmt.
  ruckruf = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  # Kompiliert das Modell mit Mean Squared Error als Verlustfunktion und Adam als Optimierer.
  # Fügt auch die Metrik Mean Absolute Error hinzu, um die Modellleistung zu überwachen.
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  # Trainiert das Modell mit den Trainingsdaten des 'window'-Objekts für eine maximale Anzahl von EPOCHS.
  # Verwendet dabei die Validierungsdaten für die Leistungsbewertung und setzt die EarlyStopping-Rückruffunktion ein.
  verlauf = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[ruckruf])

  # Gibt die Trainingshistorie zurück, die Informationen über den Verlauf des Trainings enthält.
  return verlauf
# Definiert die Anzahl der Schritte, die im Output-Fenster berücksichtigt werden sollen.
OUT_STEPS = 24  # Definiert die Länge des Vorhersagefensters.

# Erstellt eine Instanz von WindowGenerator.
# Diese Instanz wird verwendet, um Datenfenster für das Training von Zeitreihenmodellen zu erzeugen.
multi_window = WindowGenerator(inputBreite=24,  # Die Breite des Eingabefensters (Anzahl der Zeitschritte).
                            labelBreite=OUT_STEPS,  # Die Breite des Label-Fensters (gleich OUT_STEPS).
                            shift=OUT_STEPS)  # Der Versatz zwischen dem Ende des Eingabefensters und dem Beginn des Label-Fensters.
 # Erstellt leere Dictionaries, um später die Leistung des Modells auf den Validierungs- und Testdatensätzen zu speichern.
validierungsDatensätze = {}
multiLeistung = {}

from IPython import display

# Definiert die maximale Anzahl von Trainingsepochen.
MAX_EPOCHS = 50

# Erstellt ein sequentielles LSTM-Modell mit TensorFlow Keras.
multi_lstm_model = tf.keras.Sequential([
    # Eine LSTM-Schicht mit 32 Einheiten. 'return_sequences=False' bedeutet, dass nur der letzte Output der Sequenz zurückgegeben wird.
    tf.keras.layers.LSTM(32, return_sequences=False),
    
    # Eine Dense-Schicht, die die Ausgabe des LSTM auf die gewünschte Größe bringt. 
    # 'OUT_STEPS*featuresAnzahl' definiert die Gesamtzahl der Ausgabeeinheiten.
    tf.keras.layers.Dense(OUT_STEPS*featuresAnzahl, kernel_initializer=tf.initializers.zeros()),

    # Eine Reshape-Schicht, um die Ausgabe in das gewünschte Format zu bringen, hier [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, featuresAnzahl])
])

# Trainiert das Modell mit der zuvor definierten Funktion 'kompilieren', die das Modell kompiliert und dann trainiert.
verlauf = kompilieren(multi_lstm_model, multi_window)

display.clear_output()
 # Bewertet das trainierte Modell auf dem Validierungsdatensatz und speichert das Ergebnis.
validierungsDatensätze['LSTM'] = multi_lstm_model.evaluate(multi_window.val)

# Bewertet das Modell auf dem Testdatensatz und speichert das Ergebnis. 'verbose=0' unterdrückt die Ausgabe während der Evaluation.
multiLeistung['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

 #multi_window.plot(multi_lstm_model)

def train(dateVal):
  def dateSlicer(dateVal):
   from datetime import timedelta
   import pandas as pd

# Your input date string
   input_date_str = dateVal

# Convert the input date string to a datetime object using pandas
   input_date = pd.to_datetime(input_date_str)

# Round the input date to the nearest hour (if your data is hourly)
   input_date = input_date.replace(minute=0, second=0)

# Compute the start date as 24 hours before the input date
   start_date = input_date - timedelta(hours=24)

# Ensure the DataFrame index is in datetime format (if not already)
   test_data.index = pd.to_datetime(test_data.index)

# Slice the data between start_date and input_date
   data_24_hours = test_data[start_date:input_date]

# Display the result
   return data_24_hours

  def dateSlicerTest(dateVal):
   from datetime import timedelta
   import pandas as pd

# Your input date string
   input_date_str = dateVal

# Convert the input date string to a datetime object using pandas
   input_date = pd.to_datetime(input_date_str)

# Round the input date to the nearest hour (if your data is hourly)
   input_date = input_date.replace(minute=0, second=0)
   test_data.index = pd.to_datetime(test_data.index)

# Compute the start date as 24 hours before the input date
   start_date = input_date + timedelta(hours=24)

# Ensure the DataFrame index is in datetime format (if not already)


# Slice the data between start_date and input_date
   data_24_actual = test_data[input_date:start_date]

# Display the result
   return data_24_actual
  hel = dateSlicer(dateVal).copy()
  he = dateSlicerTest(dateVal).copy()
  new_Data_test = dateSlicerTest(dateVal)
  new_Data = dateSlicer(dateVal) 
  new_Data = hel.copy()
 # Auswählen der Features, die für die Prognose verwendet werden sollen.


  prognoseFeatures = ["Electricity_generation", "Hour", "Month", "Jahreszeit"]
#normalisiertDaten = new_Data.iloc[1:, :]
  hel = hel.iloc[:24]
  he = he.iloc[:24]

#print(normalisiertDaten.shape)
# Make sure that new_Data has the correct number of rows (24) and correct columns
  new_Data = new_Data.iloc[:24]  # Select only the first 24 rows


# Make sure you select the correct features before normalizing
  normalisiertDaten = (new_Data[prognoseFeatures] - train_mean[prognoseFeatures]) / train_std[prognoseFeatures]


# Now reshape should work, since normalisiertDaten should have a shape of (24, 4)
  reshape = normalisiertDaten.values.reshape(1, 24, len(prognoseFeatures))


# Continue with the prediction
  prognose = multi_lstm_model.predict(reshape)

# Normalisiert die ausgewählten Features in 'new_Data' mithilfe der Mittelwerte (train_mean) und Standardabweichungen (train_std) der Trainingsdaten.
# Dies stellt sicher, dass die Daten in einer ähnlichen Skala wie die Trainingsdaten vorliegen.
#normalisiertDaten = (new_Data[prognoseFeatures] - train_mean[prognoseFeatures]) / train_std[prognoseFeatures]

# Reshape der normalisierten Daten in das Format, das vom LSTM-Modell erwartet wird.
# Die Form ist [Anzahl der Beispiele, Zeitfenstergröße, Anzahl der Features].
#reshape = normalisiertDaten.values.reshape(1, 24, len(prognoseFeatures))

# Verwendet das trainierte LSTM-Modell, um eine Prognose basierend auf den reshaped normalisierten Daten zu erstellen.
#prognose = multi_lstm_model.predict(reshape)

#print(reshape)

# Füllt vorwärtsgerichtete fehlende Werte (NaNs) in 'new_Data' mit dem vorhergehenden gültigen Wert auf.
#new_Data.ffill(inplace=True)

# Füllt rückwärtsgerichtete fehlende Werte in 'new_Data' mit dem nachfolgenden gültigen Wert auf.

#new_Data.bfill(inplace=True)

# Überprüft, ob es immer noch irgendwelche fehlenden Werte in 'new_Data' gibt.
#if new_Data.isnull().any().any():
    # Berechnet den globalen Durchschnittswert für jeden Spalte in 'complete_data', falls noch NaNs vorhanden sind.
 #   global_mean = complete_data.mean()
    # Füllt die verbleibenden fehlenden Werte in 'new_Data' mit dem globalen Durchschnittswert.
  #  new_Data.fillna(global_mean, inplace=True)

# Überprüft erneut auf fehlende Werte. Gibt eine Nachricht aus, je nachdem, ob NaNs gefunden wurden oder nicht.
#if new_Data.isnull().any().any():
 #   print("NaNs da.")
#else:
  #  print("NaNs weg")
 # Wendet die Denormalisierungsformel an.
  index_ElectricityGeneration = prognoseFeatures.index('Electricity_generation')


  denormalisiert = prognose[:, :, index_ElectricityGeneration] * train_std['Electricity_generation'] + train_mean['Electricity_generation']

# Ausgabe zur Information, dass nun denormalisierte Vorhersagen für die Stromerzeugung folgen.


# Ersetzt negative Vorhersagewerte durch 0. Negative Werte sind in diesem Kontext (Stromerzeugung) 
# nicht sinnvoll, daher werden sie auf 0 gesetzt, um realistische Vorhersagen zu gewährleisten.
  for i in range(len(denormalisiert)):
    for j in range(len(denormalisiert[i])):
        if denormalisiert[i][j] < 0:
            denormalisiert[i][j] = 0

# Druckt die denormalisierten Prognosedaten aus. Diese Daten repräsentieren die Vorhersagen 
# des Modells für die Stromerzeugung in der ursprünglichen Skala der Daten.

  import matplotlib.pyplot as plt
 

# Tell Matplotlib to use the event handler function




# Assuming 'denormalisiert' is a 2D numpy array with shape (1, 24) for 24 hour predictions
# and 'he_Data' is a DataFrame with 'Electricity_generation' column.

  with plt.style.context('dark_background'):
  # Your subplot configuration values
   left   = 0.092 
   bottom = 0.36   
   right  = 0.597 
   top    = 0.88  
   wspace = 0.2    
   hspace = 0.2    

# Apply the configuration to all subplots
  
   x_values = he.index
   y_values = hel.index
   plt.figure(figsize=(25,11))
   plt.plot(y_values , hel['Electricity_generation'].values, marker='o', color='b', label='Input Werte')
   plt.plot(x_values , he['Electricity_generation'].values, marker='o', color='r', label='Echte Werte')
   plt.plot(x_values  ,denormalisiert.flatten(), marker='o', color='c', label='Vorhersage')


   plt.title(f"Prognose ab dem {he.index.date.max()}" )
   plt.xlabel('Stunden')
   plt.ylabel('Stromerzeugung in MW')
   plt.legend()
   plt.savefig('rnn.png')
   mng = plt.get_current_fig_manager()
   mng.full_screen_toggle()
   plt.gcf().canvas.mpl_connect('key_press_event', close_event)
   plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
   plt.show()


 





 


   
   
   



























root.mainloop()
