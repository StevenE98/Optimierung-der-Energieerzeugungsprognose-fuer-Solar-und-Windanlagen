import tkinter as tk
from tkinter import PhotoImage, Entry, Button, Label, ttk   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from IPython import display
import calendar
from datetime import datetime
from tkcalendar import Calendar
import threading    


root = tk.Tk()
root.title("Solarenergie Prognose")

root.geometry("1920x1080") #Standard-Größe
root.minsize(width=400, height=400) #kleinste Fenster
root.maxsize(width=1920, height=1080) #größte Fenster
root.resizable(width= False, height= False)

"""
label1 = tk.Label(root, text="Solarenergie Prognose", bg="green")
label1.pack(side="top", expand= True  ,fill="x")
"""

background_image = PhotoImage(file= "brend.png").zoom(x=3,y=3)
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)



"""
labeloutP = tk.Label(root , text="Output Parameter in Stunden: ",width= 30)
labeloutP.pack (side="left",padx=20, pady= 20)


text1 = tk.Text(root, height= 1 , width= 10)
text1.pack(side="left", padx=20, pady=20)
"""
def ki(offset, date):
    files = [ 
        "Air_temperature2015.csv",
        "Air_temperature2016.csv",
        "Air_temperature2017.csv",
        "Air_temperature2018.csv",
        "Air_temperature2019.csv",
        "Air_temperature2020.csv",
        "Air_temperature2021.csv",
        "Air_temperature2022.csv",
        "Diffuse_solar_radiation2015.csv",
        "Diffuse_solar_radiation2016.csv",
        "Diffuse_solar_radiation2017.csv",
        "Diffuse_solar_radiation2018.csv",
        "Diffuse_solar_radiation2019.csv",
        "Diffuse_solar_radiation2020.csv",
        "Diffuse_solar_radiation2021.csv",
        "Diffuse_solar_radiation2022.csv",
        "Relative_humidity2015.csv",
        "Relative_humidity2016.csv",
        "Relative_humidity2017.csv",
        "Relative_humidity2018.csv",
        "Relative_humidity2019.csv",
        "Relative_humidity2020.csv",
        "Relative_humidity2021.csv",
        "Relative_humidity2022.csv",
        "global_solar_radiation2015.csv",
        "global_solar_radiation2016.csv",
        "global_solar_radiation2017.csv",
        "global_solar_radiation2018.csv",
        "global_solar_radiation2019.csv",
        "global_solar_radiation2020.csv",
        "global_solar_radiation2021.csv",
        "global_solar_radiation2022.csv",]

    #files elect hingegen hat Minuten angaben was dazu führt das leere rows entsthen, mit NaN values , um dagegen zu wirken habe ich die Zeit auf Stunden gerundet
    #und die einzelnen values wie humidity , solar_radiation nach dem Datum sortiert
    files_elect = [ "electricity_generation_in_Germany_in_2015.csv",
        "electricity_generation_in_Germany_in_2016.csv",
        "electricity_generation_in_Germany_in_2017.csv",
        "electricity_generation_in_Germany_in_2018.csv",
        "electricity_generation_in_Germany_in_2019.csv",
        "electricity_generation_in_Germany_in_2020.csv",
        "electricity_generation_in_Germany_in_2021.csv",
        "electricity_generation_in_Germany_in_2022.csv"    ]


    # Definiert eine Funktion, um eine Liste von csv Dateien für eine einzelne Variable zu verarbeiten
    def process_df(files, variable_name,date):

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
            #variable_df = variable_df[(variable_df.index.month == date.month) & (variable_df.index.day == date.day)]

        
        # Nach dem Zusammenführen aller Dateien berechnet den Mittelwert über die Spalten, um sie zu konkatenieren.
        variable_df = variable_df.mean(axis=1)
        # Gibt die zusammengefassten Daten für die Variable als DataFrame zurück.
        return variable_df.to_frame(name=variable_name)
    
    
    # Verarbeitet die Daten jeder Variablen, indem die Dateinamen gefiltert und die Funktion aufgerufen wird.
    temperature_df = process_df([f for f in files if "Air_temperature" in f], "Air_temperature",date)
    diffuse_solar_radiation_df = process_df([f for f in files if "Diffuse_solar_radiation" in f], "Diffuse_solar_radiation",date)
    global_solar_radiation_df = process_df([f for f in files if "global_solar_radiation" in f], "Global_solar_radiation",date)
    humidity_df = process_df([f for f in files if "Relative_humidity" in f], "Relative_humidity",date)
    electricity_df = process_df(files_elect, "Electricity_generation",date)
    
    
    # Kombiniert die verarbeiteten DataFrames für jede Variable zu einem einzigen vollständigen DataFrame.
    complete_data = pd.concat([temperature_df, diffuse_solar_radiation_df, global_solar_radiation_df, humidity_df, electricity_df], axis=1)

    complete_data['Hour'] = complete_data.index.hour
    complete_data['Month'] = complete_data.index.month
    
    def add_season_column(df):
        # Create a new column 'Season' and initialize it with 'Winter'
        # This also handles December from the previous year
        df['Jahreszeit'] = 'Winter'
    
        # Define the seasons based on the month
        df.loc[df.index.month.isin([3, 4, 5]), 'Jahreszeit'] = 'Frühling'
        df.loc[df.index.month.isin([6, 7, 8]), 'Jahreszeit'] = 'Sommer'
        df.loc[df.index.month.isin([9, 10, 11]), 'Jahreszeit'] = 'Herbst'
        
        return df

    complete_data_with_seasons = add_season_column(complete_data.copy())
    # Apply the function to add the 'Season' column to the com

    # Zeigt den endgültigen kombinierten DataFrame mit allen Variablen an.
    complete_data = complete_data_with_seasons


    season_to_numeric = {
        'Winter': 0,
        'Frühling': 1,
        'Sommer': 2,
        'Herbst': 3
    }
    
    def get_jahreszeit(date):
        month = date.month
        if 1 <= month <= 2 or month == 12:
            return 0
        elif 3 <= month <= 5:
            return 1
        elif 6 <= month <= 8:
            return 2
        elif 9 <= month <= 11:
            return 3
    # Replace the 'Jahreszeit' column with numeric values4
    
    complete_data['Jahreszeit'] = complete_data['Jahreszeit'].replace(season_to_numeric)
    
    # Now the 'Jahreszeit' column has numeric values that represent seasons
    # Show the DataFrame to confirm the changes


    complete_data

    new_Data = complete_data.head(48)

    new_Data
    # Berechnung der Korrelationsmatrix
    matrix = complete_data.corr()

    # Verwenden von Seaborn, um eine Heatmap der Korrelationsmatrix zu erstellen
    plt.figure(figsize=(10, 8))  
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': .5})


    plt.title('Heatmap der Korrelation Features Solaranlagen')  

    # Anzeige des Plots
    plt.savefig('heatmap.png')

    complete_data.describe().transpose() # Überprüfung ob Korrektur vorgenommen werden muss an den Values in den Columns , die Daten scheinen 
    # keine besonderen Auffäligkeiten , jedoch ist Diffuse- und Global_solar_radiation eins eins die selbe csv file.

    #Das Duplikat wird entfernt
    complete_data.describe().transpose() # Überprüfung von Korrektur vorgenommen werden muss an den Values in den Columns , die Datei
    # keine besonderen Auffälligkeiten , jedoch ist Diffuse- und Global_solar_radiation eins die selbe csv file.

    # Das Duplikat wird entfernt

    if 'Global_solar_radiation' in complete_data.columns:
        complete_data = complete_data.drop('Global_solar_radiation', axis=1)
    else:
        print("Column does not exist in DataFrame")

    if 'Wind_Speed' in complete_data.columns:
        complete_data = complete_data.drop('Wind_Speed', axis=1)
    else:
        print("Column does not exist in DataFrame")

    complete_data
    # Bestimmung der Länge des kompletten Datensatzes
    n = len(complete_data)

    # Aufteilung des Datensatzes in Training, Validierung und Test
    # 70% der Daten für das Training, 20% für die Validierung, 10% für den Test
    train_df = complete_data[0:int(n*0.7)]
    val_df = complete_data[int(n*0.7):int(n*0.9)]
    test_df = complete_data[int(n*0.9):]

    # Berechnung des Mittelwerts und der Standardabweichung des Trainingsdatensatzes
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Anzahl der Features (Spalten) im Datensatz
    num_features = complete_data.shape[1]

    # Normalisierung der Trainings- Validierungs- und Testdaten
    # Durch Subtraktion des Mittelwerts und Division durch die Standardabweichung
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # Normalisierung des gesamten Datensatzes für die Visualisierung
    df_std = (complete_data - train_mean) / train_std

    df_std = df_std.melt(var_name='Column', value_name='Normalized')

    # Erstellung einer Violinplot-Visualisierung
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(complete_data.keys(), rotation=90)

    class WindowGenerator():
        def __init__(self, input_width, label_width, shift,
                    train_df=train_df, val_df=val_df, test_df=test_df,
                    label_columns=None):
            # Store the raw data.
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift
            
            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])

    w1 = WindowGenerator(
        input_width=24,
        label_width=1,
        shift=24,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=['Electricity_generation']
    )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    WindowGenerator.split_window = split_window
    
    
    example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                            np.array(train_df[100:100+w1.total_window_size]),
                            np.array(train_df[200:200+w1.total_window_size])])

    example_inputs, example_labels = w1.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')

    w1.example = example_inputs, example_labels

    def plot(self, model=None, plot_col='Electricity_generation', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    WindowGenerator.plot = plot

    w1.plot()
    plt.savefig('preview.png')

    for datenRahmen in [train_df, val_df, test_df]:
        datenRahmen.fillna(datenRahmen.mean(), inplace=True)

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds

    WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        #Get and cache an example batch of `inputs, labels` for plotting.
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example

    w1.train.element_spec

    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        label_columns=['Electricity_generation'])
    single_step_window


    class Baseline(tf.keras.Model):
        def __init__(self, label_index=None):
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]


    column_indices = {name: i for i, name in enumerate(train_df.columns)}
    baseline = Baseline(label_index=column_indices['Electricity_generation'])

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)


    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        label_columns=['Electricity_generation'])

    wide_window
    wide_window.plot(baseline)


# offset
    OUT_STEPS = offset
    multi_window = WindowGenerator(input_width=OUT_STEPS,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS)

    multi_window.plot()
    multi_window


    class MultiStepLastBaseline(tf.keras.Model):
        def call(self, inputs):
            return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    multi_val_performance = {}
    multi_performance = {}


    class RepeatBaseline(tf.keras.Model):
        def call(self, inputs):
            return inputs

    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
    multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
    multi_window.plot(repeat_baseline)


    for df in [train_df, val_df, test_df]:
        df.fillna(df.mean(), inplace=True)


    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])


    print('Input shape:', single_step_window.example[0].shape)
    print('Output shape:', linear(single_step_window.example[0]).shape)

    MAX_EPOCHS = 1
    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=patience,
                                                        mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
        return history

    history = compile_and_fit(linear, single_step_window)

    val_performance['Linear'] = linear.evaluate(single_step_window.val)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    wide_window
    wide_window.plot(linear)

    plt.bar(x = range(len(train_df.columns)),
            height=linear.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    _ = axis.set_xticklabels(train_df.columns, rotation=90)

    MAX_EPOCHS =  2
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_lstm_model, multi_window)

    display.clear_output()
    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model)
    
    #complete_data = pd.DataFrame(index=3, columns=['Month'])
    
        
    new_Data = complete_data.head(24).copy()
    new_Data.ffill(inplace=True)
    new_Data.bfill(inplace=True)
    if new_Data.isnull().any().any():
        global_mean = complete_data.mean()
        new_Data.fillna(global_mean, inplace=True)

    # Check again for NaNs
    if new_Data.isnull().any().any():
        print("NaNs are still present after forward and backward fill.")
    else:
        print("NaNs have been handled in new_Data.")

    print(new_Data.head(24))

    features_for_prediction = ["Air_temperature", "Diffuse_solar_radiation", "Relative_humidity", "Electricity_generation", "Hour", "Month", "Jahreszeit"]
    train_mean_selected = train_mean[features_for_prediction]
    train_std_selected = train_std[features_for_prediction]
    normalized_new_Data = (new_Data[features_for_prediction] - train_mean_selected) / train_std_selected

    num_features_model_expects = len(features_for_prediction)
    reshaped_data = normalized_new_Data.values.reshape(1, 24, num_features_model_expects)
    predictions = multi_lstm_model.predict(reshaped_data)
    print(reshaped_data)

    electricity_generation_index = features_for_prediction.index('Electricity_generation')
    electricity_generation_predictions = predictions[:, :, electricity_generation_index]
    first_step_electricity_generation_prediction = electricity_generation_predictions[:, 0]


    # Ensure we're working with scalar values for mean and standard deviation
    mean_electricity_generation = train_mean_selected['Electricity_generation'] # Adjust if necessary
    std_electricity_generation = train_std_selected['Electricity_generation']    # Adjust if necessary

    # Now apply the corrected denormalization formula
    denormalized_predictions = (electricity_generation_predictions * train_std_selected['Electricity_generation'])  + train_mean_selected['Electricity_generation']
    
    print("Denormalized predictions for Electricity Generation:")


    for i in range(len(denormalized_predictions)):
        for j in range(len(denormalized_predictions[i])):           
            if denormalized_predictions[i][j] < 0:
                denormalized_predictions[i][j] = 0
    
    
    valDate = new_Data.index[-1].date()
    valHour = new_Data.index[-1].hour
    
    
    print(denormalized_predictions)
    plt.close('all')
    dates_times = new_Data.index
    dates = new_Data.index.date
    hours = new_Data.index.hour
    with plt.style.context('dark_background'):
        plt.plot(denormalized_predictions.flatten(), marker='o', color = 'c')
    plt.xlabel('Stunden', color = 'r')
    plt.title('Prognose ab dem  ' + str (valDate ) + '-' + str(valHour) + ':00  '  , color = 'c' )
    plt.ylabel('Stromerzeugung  in MW' , color = 'r')
    plt.savefig('rnn.png')
    multi_lstm_model.save('lstm')

def button_click(): #das macht der Knopf
    """
    print('scale value:', round(scale.get()))
    
    wert = 1
    if round(scale.get()) != 0:
        wert = round(scale.get())
    """
    cal_date = cal.get_date()
    cal_date = datetime.strptime(cal_date, "%m/%d/%y")
    ki(24, cal_date)
    display_image = PhotoImage(file='rnn.png').zoom(x=2,y=2)
    
    # Bild im Label-Widget anzeigen
    image_label.config(image=display_image)
    image_label.image = display_image  # Referenz behalten, um Garbage Collection zu vermeiden

"""
def on_scale_change(event):
    scale_value_label.config(text=f'Interval: {round(scale.get())} std')
"""
style = ttk.Style()
style.configure('SunAndSky.TButton', 
                font=('Helvetica', 17),
                foreground='dark blue',
                background='turquoise',
                borderwidth=2,
                focusthickness=3,
                focuscolor='none'
                )
style.map('SunAndSky.TButton',
        background=[('active', 'sky blue')],
        foreground=[('active', 'blue')])

style2 = ttk.Style()
style2.configure('custom.TButton', 
                font=('Helvetica', 17),
                foreground='dark blue',  # Textfarbe
                background='turquoise',  # Hintergrundfarbe
                borderwidth=2,
                relief='raised',
                focusthickness=3) 

"""
scale = ttk.Scale(root, from_=1, to=720, orient='horizontal', command=on_scale_change, length= 250)
scale.pack(side = "left", padx=20 )

scale_value_label = ttk.Label(root, text='Interval: 1 std', style='custom.TButton')
scale_value_label.pack(side= "left")
"""

image_label = Label(root)
image_label.pack(side="right", padx=20, pady=20)

buttonL = ttk.Button(root, text="Start", command=button_click, style='SunAndSky.TButton' ) # die ganze Knöpfe
buttonL.place(width=400, height= 150)
buttonL.pack(side= "left", padx= 10)


cal = Calendar(root, selectmode='day')
cal.pack(side= "left")




root.mainloop()
