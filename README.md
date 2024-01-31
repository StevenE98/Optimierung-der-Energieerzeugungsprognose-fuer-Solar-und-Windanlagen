# Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen


Das Solarprognose Programm nutzt Wetterdaten des ERA5 Modells und Energieerzeugungsdaten des Frauenhofer-Institut für Solare Energiesysteme, um dann mit Hilfe von LSTM (long short term memory) eine möglichst genaue Energieerzeugungsprognose für sowohl Solar- als auch Windenergie zu erzeugen.

### Voraussetzungen

```
Python 3.9
Jupyter
Tensorflow
Pandas
Matlib
Seaborn
Sklearn
```

1. **Abschnitt exportierung der Klima Daten.**

   <img width="928" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/776442f6-f87e-4ca7-86cf-504d3f944a61">
   
   
> [!NOTE]
> Auf dieser Website befinden sich Klima Daten die eine wichtige Rolle spielen für die prognostizierung der Solar- und Windkraftanlagen.
> Unter folgendem **Link** [Klima Daten](https://www.energy-charts.info/charts/climate_hours/chart.htm?l=en&c=DE&source=solar_globe&year=2022&interval=year) 
> kann man auf die Website zugreifen.

<img width="184" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/df3f1868-caf1-4d6b-ba38-05685e39b09c">


   > [!IMPORTANT]
   > Bei Dateselection sollte unter Interval das Jahr angegeben werden und das zugehörige Jahr für das man die Daten exportieren will.

<img width="185" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/bbf09761-806b-4332-873e-ab0d5c582e48">
  
   
   > [!IMPORTANT]
   > Bei Climate Variables kann man die Klima Daten auswählen.

   > [!NOTE]
   > **Für die Prognose sind folgende Daten wichtig.**
   >
   > **Solaranlagen**
   > - Global Solar Radiation
   > - Air Temperature
   > - Relative Humidity
   >
   > **Windkraftanlagen**
   > - Global Solar Radiation
   > - Air Temperature
   > - Relative Humidity
   > - Wind Speed

 <img width="184" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/7e1096d0-5f01-457d-9142-897f5c5eb74e">


   > [!IMPORTANT]
   > Für die exportierung der Daten muss der file type als CSV angegeben werden.

   > [!CAUTION]
   > Daten können nur ab dem Jahr 2015 exportiert werden.

2. **Abschnitt exportierung der Energieerzeugungsdaten.**

   <img width="936" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/32642c53-643a-4a24-a2fd-3bb6101ed515">

   
   
> [!NOTE]
> Auf dieser Website befinden sich Energieerzeugungsdaten von  Solar- und Windkraftanlagen.
> Unter folgendem **Link** [Energieerzeugungsdaten](https://www.energy-charts.info/charts/power/chart.htm?l=en&c=DE&legendItems=000000000000000100000&interval=year&year=2022)
> kann man auf die Website zugreifen.

<img width="184" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/df3f1868-caf1-4d6b-ba38-05685e39b09c">


   > [!IMPORTANT]
   > Bei Dateselection sollte unter Interval das Jahr angegeben werden und das zugehörige Jahr für das man die Daten exportieren will.

<img width="190" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/f7c57fe0-d5a8-4b54-9f35-c9503aa77bd2">

  
   
   > [!IMPORTANT]
   > Bei Sources muss public ausgewählt werden.

<img width="526" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/e367ffba-257c-4bcc-b8c4-9f8af6d0fcb3">


   > [!NOTE]
   > **Für die Prognose sind folgende Daten wichtig.** Diese können durch das anklicken ausgewählt werden unter dem Chart.
   >
   > **Solaranlagen**
   > - Solar
   >
   > **Windkraftanlagen**
   > - Wind offshore
   > - Wind onshore


 <img width="184" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/7e1096d0-5f01-457d-9142-897f5c5eb74e">


   > [!IMPORTANT]
   > Für die exportierung der Daten muss der file type als CSV angegeben werden.

   > [!CAUTION]
   > Daten können nur ab dem Jahr 2015 exportiert werden.

3.1 **Abschnitt Daten Formatierung von Klimadaten.**
 
> [!NOTE]
> Die Daten aus Abschnitt 1 sollten exportiert sein damit sie in das folgende Programm eingelesen werden können.

> [!IMPORTANT]
> Der Code unter dem [Link](https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/blob/main/DatenFormatierungCSV.py)
> dient dem Zweck die exportierten CSV Daten zu überschreiben mit korrekten Datumsformat und der Spalte Value eine
> Identifizierbaren Namen zu geben der sich von den anderen CSV Dateien unterscheidet die andere Klima Daten reprsäsentieren.

> [!IMPORTANT]
> Veränderung werden nur in der Main vorgenommen , die Kommentare erklären den Vorgang.
```
 if __name__ == '__main__':
    csv = [  # Eigenen Pfad angeben zu den Klima Daten nach Zeit ordnen
      "C:\\Users\\Nikita\\Downloads\\energy-charts_Diffuse_solar_radiation_in_Germany_in_2023.csv",
        "C:\\Users\\Nikita\\Downloads\\energy-charts_Diffuse_solar_radiation_in_Germany_in_2024.csv"]   
    minJahr = 2023 # Minimum Jahr angeben von den files
    name = "Diffuse_solar_radiation" # Namen zuweisen für value und für die Bestimmung des Dateinamen
    newDF = csvChange(csv, name)
    save_data(newDF, "C:\\Users\\Nikita\\Desktop\\Projekt", name,minJahr)
    #Eigenen Pfad angeben wo die Daten abgespeichert werden sollen
```


3.2 **Abschnitt Daten Formatierung von Energierzeugungsdaten.**
> [!NOTE]
> Die Daten aus Abschnitt 2 sollten exportiert sein damit sie in das folgende Programm eingelesen werden können.

>[!IMPORTANT]
> Der Code zur Formatierung von Energierzeugungsdaten ist unter dem [Link verfügbar](https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/blob/main/Datenformatierung_Energieerzeugungsdaten.py)

> [!NOTE]
>Der Code dient dem Zweck die Daten nach Stunden zu gruppieren und den Durchschnitt zu berechnen,
>damit die Daten in einem Dataframe mit den Klimadaten vereinheitlicht werden können.

>[!Important]
> Veränderung müssen nur in der main vorgenommen werden , die Kommentare erklären die Schritte.


```
if __name__ == '__main__':
    file_electricity = 'C:\\Users\\Nikita\\Downloads\\energy-charts_Public_net_electricity_generation_in_Germany_in_2024 
    (1).csv'  #Eigenen Pfad angeben von nur einem File
    Electrictyname = 'Solar' # Namen zuweisen in abhängikeit ob es Solar oder Windkraftanlagen sind
    AngabeMW = 'Average_Solar_Power(MW)' # Namen zuweisen in abhängikeit ob es Solar oder Windkraftanlagen sind
    df_electricity_stunde = process_electricity_data(file_electricity,    Electrictyname, AngabeMW)
    df_electricity_stunde.to_csv(f"C:\\Users\\Nikita\\Desktop\\Projekt\\wind_electricity_generation
    _in_Germany_in_2024.csv", index=False)   #  Eigenen Pfad angeben zum speichern der Datei
  
```
<img width="423" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/16e71b1a-16d1-4394-a26a-acd5b880a565">

>[!NOTE]
> Der letzte Schritt besteht darin die Daten hochzuladen im Jupyter
> Lab und  sie einzulesen im Dataframe unter files .Dannach kann man
> die Prognosen inerhalb des JupyterLabs machen(Kommentare sind vorhanden).

4 **LSTM Modell (long short-term memory).**
> [!note]
> Bei der Anwendung eines LSTM-Modells für Zeitreihendaten werden die Daten zunächst in Trainings-, Validierungs- und Testsets aufgeteilt und normalisiert. Nach der Definition von Zeitfenstern wird das LSTM-Modell mit geeigneten Funktionen für Optimierung, Verlust und Leistungsmetriken kompiliert. Das Training erfolgt über model.fit() unter Einsatz der Trainings- und Validierungsdaten. Abschließend wird das Modell mit dem Testdatensatz evaluiert, um die Leistung des Modells zu beurteilen. Die Ergebnisse werden denormalisiert, um sie interpretierbar zu machen.


```

MAX_EPOCHS = 8
#gibt an wie viel Batches die Modell durchlaufen soll
multi_lstm_model = tf.keras.Sequential([
   # Eine LSTM-Schicht mit 32 Einheiten. 'return_sequences=False' bedeutet, dass nur der letzte Output der Sequenz zurückgegeben wird.
    tf.keras.layers.LSTM(32, return_sequences=False),
    
    tf.keras.layers.Dense(OUT_STEPS*featuresAnzahl, kernel_initializer=tf.initializers.zeros()),

    # Eine Reshape-Schicht, um die Ausgabe in das gewünschte Format zu bringen, hier [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, featuresAnzahl])
   ])
verlauf = kompilieren(multi_lstm_model, multi_window)

display.clear_output()

validierungsDatensätze['LSTM'] = multi_lstm_model.evaluate(multi_window.val)

multiLeistung['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

multi_window.plot(multi_lstm_model)

```

![Screenshot 2024-01-15 141017](https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/127848322/6c7c3e38-e899-4cc3-814d-1220c3e6399e)

5 **GUI**
Dieses GUI wurde entwickelt, um Vorhersagen für die Solarerzeugungsprognose basierend auf einem tranierten LSTM-Modell zu erstellen. Die GUI verwendet Tkinter und ttkbootstrap für die Benutzeroberfläche. In GUI wird die LSTM auf 50 Epoche gesetzt und nur die "electricty generation in Germany" csv Dateien benutzt um die Programm schneller zu starten

<img width="890" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/127848322/ac80b136-610a-4faa-9818-dc21b7d058d1">
 
>[!IMPORTANT]
Stellen Sie sicher, dass Sie die folgenden Bibliotheken installiert haben: $pip install tkinter ttkbootstrap numpy pandas matplotlib seaborn scikit-learn pillow



1. GUI starten: in Visual Studio Code in jupyter lab entstehen Schwierigkeiten.
2. Das Programm ausführen , warten bis alle Epochen durchlaufen und das Modell endgültig trainiert ist .
3. Ein bestimmtes Datum und eine bestimmte Uhrzeit für die Vorhersage in Kalender festlegen , bis 2023 da die Werte mit aktuellen Daten verglichen werden können.
4. Werte von 2024 können hinzugefügt werden wenn sie vorhanden sind.
5. Klicken Sie auf "Datum festlegen", um das Datum einzustellen
6. Klicken Sie auf "Start", um die Prognose zu starten
7. Auf der grafischen Benutzeroberfläche wird ein Diagramm mit Eingabewerten, wahren Werten und vorhergesagten Werten angezeigt.
8. Drücken Sie die "Escape" um zum Datumeingabefenster zurückzukehren

![Screenshot 2024-01-31 163433](https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/127848322/0049e94e-7ee3-43d1-906c-8d28b6c9d8af)


5 **Conclusion.**

>[!IMPORTANT]
> Solarprognosen können erflogreich durchgeführt werden. 
> Die prozentuale Abweichung unter Verwendung des LSTM-Modells für Solaranlagen liegt bei etwa 11,52% und für Windkraftanlagen bei 17,95% .
> Die Prognose schwankt jedoch stark wenn der Zeitraum weiter in der Zukunft liegt.
> Wenn nur die Electricity generation Daten verwendet werden und keine Klima Daten konnte sogar die Abweichung von 0.0781 % erreicht werden.![image](https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/c10fc56b-5a5c-4b07-9106-42eeb1d68b8f)

>


>[!NOTE]
> Für ein genaueres Ergebnis der Prognose kann noch ein Autoencoder hinzugefügt werden.
> Mit diesem können dann noch beispielsweise Vorhersagen von Wetterdiensten verwendet werden, 
> wodurch die entstehende Prognose weiter angepasst werden kann.
> Die Implimentation des Autoencoderes kann wegen des jetztigen Zeitrahmens zur Zeit nicht realisiert werden.



>[!IMPORTANT]
> Die Vorhersage von Windkraftanlagen-Prognose ist im Gegensatz zur Solar-Prognosen schwieriger zu bestimmen



## Autoren

* **Nikita Masch**
* **Steven Edy**
* **Julian Walter**
