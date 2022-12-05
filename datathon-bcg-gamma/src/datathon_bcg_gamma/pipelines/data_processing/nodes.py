import pandas as pd 
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder


def add_days (data_road_row :pd.DataFrame) -> pd.DataFrame:
    data_road_row['Date et heure de comptage'] = pd.to_datetime(data_road_row['Date et heure de comptage'], utc=True)
    data_road_row['date'] = data_road_row['Date et heure de comptage'].apply(lambda x : x.date())
    data_road_row['hour'] = data_road_row['Date et heure de comptage'].apply(lambda x : x.hour)
    data_road_row['day'] = data_road_row['date'].apply(lambda x : x.day)
    data_road_row['month'] = data_road_row['date'].apply(lambda x : x.month)
    data_road_row['year'] = data_road_row['date'].apply(lambda x : x.year)
    data_road_row['day_of_week'] = data_road_row['Date et heure de comptage'].apply(lambda x : x.day_name())
    cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cat_type = CategoricalDtype(categories=cats, ordered=True)
    data_road_row['day_of_week'] = data_road_row['day_of_week'].astype(cat_type)
    lb = LabelEncoder()
    # data_road_row['day_of_week_2'] = lb.fit_transform( data_road_row['day_of_week'] )
    data_road_row = data_road_row.sort_values('Date et heure de comptage')
    data_road_row['vacances'] = 0
    mask_1 = (data_road_row['Date et heure de comptage'] > "2021-12-18") & (data_road_row['Date et heure de comptage'] <= "2022-01-04")
    mask_2 = (data_road_row['Date et heure de comptage'] > "2022-02-19") & (data_road_row['Date et heure de comptage'] <= "2022-03-06")
    mask_3 = (data_road_row['Date et heure de comptage'] > "2022-04-23") & (data_road_row['Date et heure de comptage'] <= "2022-05-10")
    mask_4 = (data_road_row['Date et heure de comptage'] > "2022-07-07") & (data_road_row['Date et heure de comptage'] <= "2022-09-02")
    mask_5 = (data_road_row['Date et heure de comptage'] > "2022-10-22") & (data_road_row['Date et heure de comptage'] <= "2022-11-08")
    data_road_row.loc[mask_1,"vacances"] = 1
    data_road_row.loc[mask_2,"vacances"] = 1
    data_road_row.loc[mask_3,"vacances"] = 1
    data_road_row.loc[mask_4,"vacances"] = 1
    data_road_row.loc[mask_5,"vacances"] = 1
    data_road_row = data_road_row.reset_index()
    data_road_row = data_road_row.drop('index',axis = 1)
    data_road_row = data_road_row.drop(columns=['Identifiant arc','Etat arc','Etat trafic','Identifiant noeud amont','Libelle noeud amont','Identifiant noeud aval','Libelle noeud aval','Date debut dispo data','Date fin dispo data','geo_point_2d','geo_shape'])
    return data_road_row

def attach_meteo(data_meteo:pd.DataFrame, df:pd.DataFrame) -> pd.DataFrame:
    data_meteo['DATE'] = pd.to_datetime(data_meteo['DATE'])
    data_meteo['date'] = data_meteo['DATE'].apply(lambda x : x.date())
    data_meteo = data_meteo.drop(columns=['DATE','HUMIDITY_MAX_PERCENT','PRESSURE_MAX_MB','CLOUDCOVER_AVG_PERCENT','HEATINDEX_MAX_C','DEWPOINT_MAX_C','WEATHER_CODE_MORNING','WEATHER_CODE_NOON','WEATHER_CODE_EVENING','OPINION','SUNSET','SUNRISE','TEMPERATURE_NIGHT_C']) 
    data_meteo = data_meteo.sort_values('date')
    data_meteo = data_meteo.reset_index()
    data_meteo = data_meteo.drop('index',axis = 1)
    df = df.merge(data_meteo, how='left',on ='date')
    return df


def add_jours_f(jours_feries : pd.DataFrame, df_road: pd.DataFrame):
    jours_feries[["year", "month", "day"]] = pd.DataFrame(jours_feries["date"].str.split("-").to_list()).astype('int')
    jours_feries.drop(columns=["date", "annee", "zone", "nom_jour_ferie"], inplace=True)
    jours_feries["est_ferie"] = 1
    df_road = df_road.merge(jours_feries, on=["year", "month", "day"], how='left')
    df_road['est_ferie'].fillna(0, inplace=True)
    df_road = pd.get_dummies(df_road, columns=['year'], prefix='year')
    df_road = pd.get_dummies(df_road, columns=['hour'], prefix='hour')
    df_road = pd.get_dummies(df_road, columns=['day'], prefix='day')
    df_road = pd.get_dummies(df_road, columns=['day_of_week'], prefix='wday')
    df_road = pd.get_dummies(df_road, columns=['month'], prefix='month')
    return df_road