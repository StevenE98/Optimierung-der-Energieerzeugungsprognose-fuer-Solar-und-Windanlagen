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
    df['Jahr'] = int(year[0]) # Füge das Jahr zum DataFrame hinzu
    df.rename(columns={'value': name}, inplace=True)

    # Gehe durch jede Zeile und ändere den Monatsnamen in einen numerischen Wert
    for index, row in df.iterrows():
        month = row['Date']
        # Konvertiere Monatsnamen in Zahlen (01 für Januar, 02 für Februar, usw.)
        month= row['Date']

        if month == "January":
            df.at[index,'Date'] = '01'
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
        # ... (gleiches für andere Monate)

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
    df['Date (GMT+1)'] = (pd.to_datetime(df['Jahr'].astype(str) + '-' + df['Date'].astype(str) + '-' + df['Tag'].astype(str) + ' ' + df['y'].astype(str) + ':00:00') + timedelta(hours=1))
    df['Date (GMT+1)'] = df['Date (GMT+1)'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Entferne nicht mehr benötigte Spalten
    df = df.drop('Tag', axis=1)
    df = df.drop('Date', axis=1)
    df = df.drop('y', axis=1)
    df = df.drop('Jahr',axis=1)

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
  
    # Konvertiere Monatsnamen in Zahlen und erstelle das Datumsformat
    months = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }

    df['Date'] = df['Date'].map(months)

    df['Date (GMT+1)'] = pd.to_datetime('2016-' + df['Date'].astype(str) + '-' + df['y'].astype(str).str.zfill(2) + 'T00:00:00') + pd.to_timedelta(df['y'], unit='h')
    df['Date (GMT+1)'] = df['Date (GMT+1)'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Entferne unnötige Spalten
    df = df.drop(['Date', 'y'], axis=1)

    return df

# Die Funktion 'save_data' speichert die bearbeiteten DataFrames in CSV-Dateien hat einen Counter für das Jahresdatum
def save_data(dataframes, base_path):
    count = 2015
    for df in dataframes:
        df.to_csv(f"{base_path}/Wind_Speed{count}.csv", index=False)
        count += 1

# Hauptteil des Skripts, in dem die Funktionen aufgerufen werden
if __name__ == '__main__':
  csv = [



        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2015.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2016.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2017.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2018.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2019.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2020.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2021.csv",
        "C:\\Users\\Masch\\Downloads\\energy-charts_Wind_speed_in_Germany_in_2022.csv"
    ]
    name = "Wind_speed"
    newDF = csvChange(csv, name)
    save_data(newDF, "C:\\Users\\Masch\\Desktop\\Projekt")
