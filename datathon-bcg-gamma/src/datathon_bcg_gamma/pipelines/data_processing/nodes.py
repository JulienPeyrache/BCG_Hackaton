import pandas as pd 
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import random



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

# def impute(data_):
#     #Define a subset of the dataset
#     data = data_.copy()
#     df_knn = data.filter(['hour','day','month','year','day_of_week_2', 'Débit horaire', "Taux d'occupation"]).copy()

#     # Define scaler to set values between 0 and 1
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     df_knn = pd.DataFrame(scaler.fit_transform(df_knn), columns = df_knn.columns)

#     # Define KNN imputer and fill missing values
#     knn_imputer = KNNImputer(n_neighbors=2, weights='uniform', metric ='nan_euclidean')
#     df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
#     inversed = pd.DataFrame(scaler.inverse_transform(df_knn_imputed))
#     data_['filled_tx_occup'] = inversed[5]
#     data_['filled_debit'] = inversed[4]

#     return data_


# def impute(data_:pd.DataFrame,n:int) -> pd.DataFrame:
#     #Define a subset of the dataset
#     name_hour = ['hour_%d' % (i) for i in range(0,24) ]
#     name_month = ['day_%d' % (i) for i in range(1,32) ]
#     data = data_.copy()
#     df_knn = data.filter(['Débit horaire', "Taux d'occupation"] + name_hour).copy()

#     # Define scaler to set values between 0 and 1
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     df_knn = pd.DataFrame(scaler.fit_transform(df_knn), columns = df_knn.columns)

#     # Define KNN imputer and fill missing values
#     knn_imputer = KNNImputer(n_neighbors=n, weights='distance', metric ='nan_euclidean')
#     df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
#     inversed = pd.DataFrame(scaler.inverse_transform(df_knn_imputed))

#     data['filled_tx_occup'] = inversed[1]
#     data['filled_debit'] = inversed[0]
#     return data

def replace_na (data) : 
    index_taux = data.loc[data["Taux d'occupation"].isna()].index
    index_debit = data.loc[data["Débit horaire"].isna()].index
    mean = data.groupby(['day_of_week', 'hour'])[["Débit horaire","Taux d'occupation"]].mean().unstack()
    for idx_t in index_debit:
        data.loc[idx_t,'Débit horaire'] = mean.loc[data.loc[idx_t,'day_of_week'], 'Débit horaire'][data.loc[idx_t,'hour']] 
    for idx_d in index_taux:
        data.loc[idx_d,"Taux d'occupation"] = mean.loc[data.loc[idx_d,'day_of_week'], "Taux d'occupation"][data.loc[idx_d,'hour']] 
    return data

# def replace_na (data :pd.DataFrame) -> pd.DataFrame:

#     data.groupby(['day_of_week', 'hour'])[['']].mean()


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