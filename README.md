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

3. **Abschnitt Daten Formatierung.**
 
> [!NOTE]
> Die Daten aus Abschnitt 1 sollten exportiert sein damit sie in das folgende Programm eingelesen werden können.

> [!IMPORTANT]
> Was nun zu beachten ist das

```
   import pandas as pd
from datetime import timedelta
import re


def csvChange(csv, name):
    listdf = []

    # Gehe durch jede Datei in der übergebenen CSV-Liste
    for files in csv:

        # Lese die CSV-Datei in ein DataFrame
        df = pd.read_csv(files)

        # Extrahiere das Jahr aus dem Dateinamen mit Regex
        year = re.findall('\d{4}', files)
        df['Tag'] = 0
        df['Jahr'] = int(year[0])  # Füge das Jahr zum DataFrame hinzu
        df.rename(columns={'value': name}, inplace=True)

        # Gehe durch jede Zeile und ändere den Monatsnamen in einen numerischen Wert
        for index, row in df.iterrows():
            month = row['Date']
            # Konvertiere Monatsnamen in Zahlen (01 für Januar, 02 für Februar, usw.)
            if month == "January":
                df.at[index, 'Date'] = '01'
            elif month == "February":
                df.at[index, 'Date'] = '02'

            elif month == "March":
                df.at[index, 'Date'] = '03'

            elif month == "April":
                df.at[index, 'Date'] = '04'

            elif month == "May":
                df.at[index, 'Date'] = '05'

            elif month == "June":
                df.at[index, 'Date'] = '06'

            elif month == "July":
                df.at[index, 'Date'] = '07'

            elif month == "August":
                df.at[index, 'Date'] = '08'

            elif month == "September":
                df.at[index, 'Date'] = '09'

            elif month == "October":
                df.at[index, 'Date'] = '10'

            elif month == "November":
                df.at[index, 'Date'] = '11'

            elif month == "December":
                df.at[index, 'Date'] = '12'

            else:
                df.at[index, 'Date'] = None

        # Bereinige und konvertiere die 'y'-Spalte in numerische Werte, entferne nicht-numerische Zeilen
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['y'])
        df['y'] = df['y'].astype(int)

        # Ermittle den Tag des Monats basierend auf der Stunde
        currentMonth = "01"
        counter = 0
        for index, row in df.iterrows():
            current_hour = row['y']
            newMonth = row['Date']
            # Zähle die Tage innerhalb des gleichen Monats
            if current_hour == 0:
                if currentMonth == newMonth:
                    counter += 1
                else:
                    counter = 1
                    currentMonth = newMonth
            df.at[index, 'Tag'] = counter

        # Erstelle ein neues Datumsformat mit Zeitzone GMT+1
        df['Date (GMT+1)'] = (pd.to_datetime(
            df['Jahr'].astype(str) + '-' + df['Date'].astype(str) + '-' + df['Tag'].astype(str) + ' ' + df['y'].astype(
                str) + ':00:00') + timedelta(hours=1))
        df['Date (GMT+1)'] = df['Date (GMT+1)'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Entferne nicht mehr benötigte Spalten
        df = df.drop('Tag', axis=1)
        df = df.drop('Date', axis=1)
        df = df.drop('y', axis=1)
        df = df.drop('Jahr', axis=1)

        # Füge das bearbeitete DataFrame zur Liste hinzu
        listdf.append(df)
    return listdf


# Die Funktion 'data_n' dient zur weiteren Datenbereinigung
def data_n(df):
    # Entferne die erste Zeile, wenn diese nicht numerisch ist
    if pd.isna(df.iloc[0]['y']) or not str(df.iloc[0]['y']).isdigit():
        df = df.drop(0)

    # Konvertiere 'y' und 'value' in numerische Werte und entferne Zeilen mit fehlenden Werten
    df['y'] = pd.to_numeric(df['y'], errors='coerce').astype(int)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['y', 'value'])
    months = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }


    df['Date'] = df['Date'].map(months)

    df['Date (GMT+1)'] = pd.to_datetime(
    '2016-' + df['Date'].astype(str) + '-' + df['y'].astype(str).str.zfill(2) + 'T00:00:00') + pd.to_timedelta(df['y'],
                                                                                                               unit='h')
    df['Date (GMT+1)'] = df['Date (GMT+1)'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    df = df.drop(['Date', 'y'], axis=1)

    return df


# Die Funktion 'save_data' speichert die bearbeiteten DataFrames in CSV-Dateien
def save_data(dataframes, base_path):
    count = 2023
    for df in dataframes:
        df.to_csv(f"{base_path}/Diffuse_solar_radiation{count}.csv", index=False)
        count += 1


# Hauptteil des Skripts, in dem die Funktionen aufgerufen werden
if __name__ == '__main__':
    csv = [
        "C:\\Users\\Nikita\\Downloads\\energy-charts_Diffuse_solar_radiation_in_Germany_in_2023.csv",
        "C:\\Users\\Nikita\\Downloads\\energy-charts_Diffuse_solar_radiation_in_Germany_in_2024.csv"
        ]
    name = "Diffuse_solar_radiation"
    newDF = csvChange(csv, name)
    save_data(newDF, "C:\\Users\\Nikita\\Desktop\\Projekt")
```
   
  
   
   

   
 





 
   







## Autoren

* **Nikita Masch**
* **Steven Edy**
* **Julian Walter**
