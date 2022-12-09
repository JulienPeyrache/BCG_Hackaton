import pandas as pd 
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import  mean_squared_error
import numpy as np
import pickle
import matplotlib.pyplot as plt



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

def add_days_input (data_road_row :pd.DataFrame) -> pd.DataFrame:
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
    for i in range(1,12):
        data_road_row[('month_%d' %(i))]=0
    for i in range(1,4):
        data_road_row[('day_%d' %(i))]=0

    ## à changer à 9
    for i in range(8,32):
        data_road_row[('day_%d' %(i))]=0
    data_road_row['year_2021']=0

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
    data_meteo = data_meteo[['date','MAX_TEMPERATURE_C','MIN_TEMPERATURE_C','WINDSPEED_MAX_KMH','PRECIP_TOTAL_DAY_MM','TOTAL_SNOW_MM','SUNHOUR']]
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




def _normalize_n(data):
    return (data - data.mean().mean())/data.std().mean()

def _unormalize_n(data, previous):
    return (data*previous.std().mean()) + previous.mean().mean()


def _input_to_supervised(data, n_in=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):
        cols.append(data[["Taux d'occupation",'Débit horaire']].shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(2)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    data = data.merge(agg, left_index=True, right_index=True)
    if dropnan:
        data.dropna(inplace=True)
    return data


##Fenetrage

def _series_to_supervised(data, n_in=1, n_out=1, dropnan=True, is_data=True):
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data[["Taux d'occupation",'Débit horaire']].shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(2)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data[["Taux d'occupation",'Débit horaire']].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(2)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(2)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    data = data.merge(agg, left_index=True, right_index=True)
    # drop rows with NaN values
    if dropnan:
        data.dropna(inplace=True)
    return data


def _create_pipeline(num_col, min_max_col):
    ##Pipelines Scikit Learn
    ct_scaler = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), num_col),
            ("scaler_minmax", MinMaxScaler(), min_max_col),
        ]
    )

    pipeline_master = Pipeline([
        ("preprocessor", ct_scaler),
        ])
    
    return pipeline_master

def prepare_input_for_model(df_model:pd.DataFrame, n_past_values:int):
    df_fenetrage = _input_to_supervised(df_model, n_past_values, dropnan=False)
    df_fenetrage = df_fenetrage.drop(
        columns=["Taux d'occupation", "Débit horaire", 'Libelle', 'Date et heure de comptage', 'date'])
    min_max_columns_n = ['vacances', 'est_ferie', 'year_2021', 'year_2022', 'hour_0', 'hour_1', 'hour_2', 'hour_3',
                         'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
                         'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20',
                         'hour_21', 'hour_22', 'hour_23', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7',
                         'day_8', 'day_9', 'day_10', 'day_11', 'day_12', 'day_13', 'day_14', 'day_15', 'day_16',
                         'day_17', 'day_18', 'day_19', 'day_20', 'day_21', 'day_22', 'day_23', 'day_24', 'day_25',
                         'day_26', 'day_27', 'day_28', 'day_29', 'day_30', 'day_31', 'wday_Monday', 'wday_Tuesday',
                         'wday_Wednesday', 'wday_Thursday', 'wday_Friday', 'wday_Saturday', 'wday_Sunday', 'month_1',
                         'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
                         'month_10', 'month_11', 'month_12']
    numerical_col_n = [feature for feature in list(df_fenetrage) if feature not in min_max_columns_n]
    pipeline_n = _create_pipeline(numerical_col_n, min_max_columns_n)
    input = pipeline_n.fit_transform(df_fenetrage)
    return input


def prepare_data_for_model_taux(df_model:pd.DataFrame, n_past_values:int,n_output:int) -> Tuple:
    df_fenetrage = _series_to_supervised(df_model,n_past_values,n_output)
    df_fenetrage = df_fenetrage.drop(columns=["Taux d'occupation","Débit horaire",'Libelle','Date et heure de comptage','date'])
    names_y = ['var1(t)']
    names_y += [('var1(t+%d)' % (i)) for i in range(1,n_output)]
    names_dropping = [('var2(t+%d)' % (i)) for i in range(1,n_output)]
    names_dropping += ['var2(t)']
    y = df_fenetrage[names_y]
    X = df_fenetrage.drop(columns=names_y+names_dropping)
    X_train , X_test ,y_train , y_test = train_test_split(X,y, test_size=0.2)
    y_train_normalize = _normalize_n(y_train)
    min_max_columns_n = ['vacances','est_ferie','year_2021','year_2022','hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7','hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21','hour_22','hour_23','day_1','day_2','day_3','day_4','day_5','day_6','day_7','day_8','day_9','day_10','day_11','day_12','day_13','day_14','day_15','day_16','day_17','day_18','day_19','day_20','day_21','day_22','day_23','day_24','day_25','day_26','day_27','day_28','day_29','day_30','day_31','wday_Monday','wday_Tuesday','wday_Wednesday','wday_Thursday','wday_Friday','wday_Saturday','wday_Sunday','month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12']
    numerical_col_n = [feature for feature in list(X) if feature not in min_max_columns_n]
    pipeline_n = _create_pipeline(numerical_col_n,min_max_columns_n)
    X_train_preprocessed = pipeline_n.fit_transform(X_train)
    X_test_preprocessed = pipeline_n.transform(X_test)
    return X_train_preprocessed, X_test_preprocessed, y_train_normalize, y_train, y_test




def prepare_data_for_model_debit(df_model:pd.DataFrame, n_past_values:int,n_output:int) -> Tuple:
    df_fenetrage = _series_to_supervised(df_model,n_past_values, n_output)
    df_fenetrage = df_fenetrage.drop(columns=["Taux d'occupation","Débit horaire",'Libelle','Date et heure de comptage','date']) 
    names_y = ['var2(t)']
    names_y += [('var2(t+%d)' % (i)) for i in range(1,n_output)]
    names_dropping  = [('var1(t+%d)' % (i)) for i in range(1,n_output)]
    names_dropping += ['var1(t)']
    y = df_fenetrage[names_y]
    X = df_fenetrage.drop(columns=names_y+names_dropping)
    X_train , X_test ,y_train , y_test = train_test_split(X,y, test_size=0.2)
    y_train_normalize = _normalize_n(y_train)
    min_max_columns_n = ['vacances','est_ferie','year_2021','year_2022','hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7','hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21','hour_22','hour_23','day_1','day_2','day_3','day_4','day_5','day_6','day_7','day_8','day_9','day_10','day_11','day_12','day_13','day_14','day_15','day_16','day_17','day_18','day_19','day_20','day_21','day_22','day_23','day_24','day_25','day_26','day_27','day_28','day_29','day_30','day_31','wday_Monday','wday_Tuesday','wday_Wednesday','wday_Thursday','wday_Friday','wday_Saturday','wday_Sunday','month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12']
    numerical_col_n = [feature for feature in list(X) if feature not in min_max_columns_n]
    pipeline_n = _create_pipeline(numerical_col_n,min_max_columns_n)
    X_train_preprocessed = pipeline_n.fit_transform(X_train)
    X_test_preprocessed = pipeline_n.transform(X_test)
    return X_train_preprocessed, X_test_preprocessed, y_train_normalize, y_train, y_test



def train_model (X_train:pd.DataFrame,y_train:pd.DataFrame, params_model:Dict):
    model = XGBRegressor(**params_model)
    model.fit(X_train, y_train)
    return model

def evaluate_models(model, X_test, y_test, y_train):
    y_pred= model.predict(X_test)
    y_pred_n = _unormalize_n(y_pred, y_train)
    MSE = mean_squared_error(y_test, y_pred_n)
    print('Normalise RMSE du model : ', MSE**0.5/(y_test.max().max() - y_test.min().min())*100,'%')
    t = np.arange(len(y_test.iloc[-1, :]))
    plt.plot(t,y_test.iloc[-1, :], 'r')
    plt.plot(t,y_pred_n[-1], 'b')
    plt.show()
    return MSE**0.5

def evaluate_models_output(model, X_test, y_train):
    y_pred= model.predict(X_test)
    y_pred_n = _unormalize_n(y_pred, y_train)
    plt.plot(y_pred_n[-1], 'b')
    plt.show()



