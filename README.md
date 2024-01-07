# Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen

Das Solarprognose Programm nutzt Wetterdaten des ERA5 Modells und Energieerzeugungsdaten des Frauenhofer-Institut für Solare Energiesysteme, um dann mit Hilfe von KI eine möglichst genaue Energieerzeugungsprognose für sowohl Solar- als auch Windenergie zu erzeugen.

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
   >[!NOTE]
   >Auf dieser Website befinden sich Klima Daten die eine wichtige Rolle spielen für die prognostizierung der Solar- und Windanlagen.
   >Unter folgendem Link [Klima Daten](https://www.energy-charts.info/charts/climate_hours/chart.htm?l=en&c=DE&source=solar_globe&year=2022&interval=year) 
   >kann man auf die Website zugreifen.

<img width="184" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/df3f1868-caf1-4d6b-ba38-05685e39b09c">


   >[!IMPORTANT]
   >Bei Dateselection sollte unter Interval das Jahr angegeben werden und das zugehörige Jahr für das man die Daten exportieren will.

<img width="185" alt="image" src="https://github.com/StevenE98/Optimierung-der-Energieerzeugungsprognose-fuer-Solar-und-Windanlagen/assets/114944673/bbf09761-806b-4332-873e-ab0d5c582e48">
   >[!IMPORTANT]
   >Bei Climate Variables kann man die Klima Daten auswählen.

   [!NOTE]
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


   [!IMPORTANT]
   > Für die exportierung der Daten muss der file type als CSV angegeben werden.

   [!Warning]
   > Daten können ab dem Jahr 2015 exportiert werden.


   
 





 
   







## Autoren

* **Nikita Masch**
* **Steven Edy**
* **Julian Walter**
