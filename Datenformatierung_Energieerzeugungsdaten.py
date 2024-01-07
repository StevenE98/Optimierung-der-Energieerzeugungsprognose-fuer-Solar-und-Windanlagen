import pandas as pd
def process_electricity_data(file_path, Electricityname, Angabemw):

    # Lese die CSV-Datei ein
    df = pd.read_csv(file_path)

    # Entfernen der ersten Zeile, falls sie keine echten Daten enthält
    df = df.iloc[1:]

    # Umwandle 'Date (GMT+1)' in datetime ohne Zeitzone
    df['Date (GMT+1)'] = pd.to_datetime(df['Date (GMT+1)'], utc=True).dt.tz_convert(None)

    # m numerische Format sicherstellen
    for col in df.columns:
        if col != 'Date (GMT+1)':  # Überspringe die Datums-Spalte
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Entferne alle Zeilen mit NaN-Werten, die während der Umwandlung entstanden sind
    df.dropna(inplace=True)

    #  zu stündlichen Daten umwandeln
    df['Hour'] = df['Date (GMT+1)'].dt.floor('H')
    df_stunde= df.groupby('Hour').mean().reset_index()

    # Spalten umschreiben
    df_stunde.rename(columns={ Electrictyname: AngabeMW}, inplace=True)

    # Datumsformat  bestimmen , um es mit den anderen Daten abgleichen zu können
    df_stunde['Date (GMT+1)'] = df_stunde['Hour'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_stunde.drop('Hour', axis=1, inplace=True)

    return df_stunde

if __name__ == '__main__':
    file_electricity_2015 = 'C:\\Users\\Nikita\\Downloads\\energy-charts_Public_net_electricity_generation_in_Germany_in_2024 (1).csv'  #Daten zum einlesen
    Electrictyname = 'Solar'
    AngabeMW = 'Average_Solar_Power(MW)'
    df_electricity_hourly = process_electricity_data(file_electricity_2015,    Electrictyname, AngabeMW)
    df_electricity_hourly.to_csv("path_to_save_processed_data.csv", index=False)
    df_electricity_hourly.to_csv(f"C:\\Users\\Nikita\\Desktop\\Projekt\\wind_electricity_generation_in_Germany_in_2024.csv", index=False)
    #  Pfad, um die Datei zu speichern
