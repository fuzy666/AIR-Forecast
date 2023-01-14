import requests
import os
import joblib
import pandas as pd
import datetime
import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from dotenv import load_dotenv
load_dotenv(override=True)


def decode_features(df, feature_view):
    """Decodes features in the input DataFrame using corresponding Hopsworks Feature Store transformation functions"""
    df_res = df.copy()

    import inspect


    td_transformation_functions = feature_view._batch_scoring_server._transformation_functions

    res = {}
    for feature_name in td_transformation_functions:
        if feature_name in df_res.columns:
            td_transformation_function = td_transformation_functions[feature_name]
            sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
            param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
            if td_transformation_function.name == "min_max_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * (param_dict["max_value"] - param_dict["min_value"]) + param_dict["min_value"])

            elif td_transformation_function.name == "standard_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * param_dict['std_dev'] + param_dict["mean"])
            elif td_transformation_function.name == "label_encoder":
                dictionary = param_dict['value_to_index']
                dictionary_ = {v: k for k, v in dictionary.items()}
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: dictionary_[x])
    return df_res


def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model.pkl"
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model.pkl")

    return model

    
def get_weather_data_weekly(city: str, start_date: datetime) -> pd.DataFrame:
    next7days_weather=pd.read_csv('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Guangzhou/next7days?unitGroup=metric&include=days&key=5WNL2M94KKQ4R4F32LFV8DPE4&contentType=csv')
########################城市名############################
    
    df_weather = pd.DataFrame(next7days_weather)
    df_weather.rename(columns = {"datetime": "date"}, 
              inplace = True)
#########################根据模型的feature进行修改###############################
    df_weather = df_weather.drop(labels=['stations','icon','description','conditions','sunset','sunrise','severerisk','preciptype','name','feelslikemax','temp','precipprob','windspeed','cloudcover','precip','tempmax','uvindex','solarradiation','solarenergy','winddir','moonphase','snow','snowdepth'], axis=1) 

    return df_weather

def get_aplevel(temps:np.ndarray) -> list:
    boundary_list = np.array([0, 50, 100, 150, 200, 300]) # assert temps.shape == [x, 1]
    redf = np.logical_not(temps<=boundary_list) # temps.shape[0] x boundary_list.shape[0] ndarray
    hift = np.concatenate((np.roll(redf, -1)[:, :-1], np.full((temps.shape[0], 1), False)), axis = 1)
    cat = np.nonzero(np.not_equal(redf,hift))

    air_pollution_level = ['Good', 'Moderate', 'Unhealthy for sensitive Groups','Unhealthy' ,'Very Unhealthy', 'Hazardous']
    level = [air_pollution_level[el] for el in cat[1]]
    return level
    
def timestamp_2_time(x):
    dt_obj = datetime.datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)
