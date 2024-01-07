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
> Bei der Anwendung eines LSTM-Modells für Zeitreihendaten werden die Daten zunächst in Trainings-, Validierungs- und Testsets aufgeteilt und normalisiert. Nach der Definition von Zeitfenstern wird das LSTM-Modell mit geeigneten Funktionen für Optimierung, Verlust und Leistungsmetriken kompiliert. Das Training erfolgt über model.fit() unter Einsatz der Trainings- und Validierungsdaten. Abschließend wird das Modell mit dem Testdatensatz evaluiert und die Ergebnisse werden denormalisiert, um die Leistung des Modells zu beurteilen.


5 **Conclusion.**
>[!IMPORTANT]
> Solarprognosen können erflogreich durchgeführt werden. 
> Die prozentuale Abweichung unter Verwendung des LSTM-Modells liegt bei etwa 18%.
> Diese schwankt jedoch stark wenn sich der Zeitraum der Prognose verändert. 

>[!NOTE]
> Für ein genaueres Ergebnis der Prognose kann noch ein Autoencoder hinzugefügt werden.
> Mit diesem können dann noch beispielsweise Vorhersagen von Wetterdiensten verwendet werden, 
> wodurch die entstehende Prognose weiter angepasst werden kann. 

>[!NOTE]
> Zum jetzigen Zeitpunkt kann der Autoencoder nicht im Modell integriert werden.

>[!IMPORTANT]
> Die Vorhersage von Windkraft ist im Gegensatz zur Solar-Prognose weitaus komplexer und auch ungenauer, 
> weshalb die Integration dieser erstmal nicht fortgesetzt wurde.






  
   
   

   
 





 
   







## Autoren

* **Nikita Masch**
* **Steven Edy**
* **Julian Walter**
