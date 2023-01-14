import streamlit as st
import hopsworks
import joblib
from datetime import date
import pandas as pd
from datetime import timedelta, datetime
from functions import *
import numpy as np
from sklearn.preprocessing import StandardScaler

import folium
from streamlit_folium import st_folium, folium_static
import json
import time
from branca.element import Figure


def fancy_header(text, font_size=24):
    res = f'<p style="color:#ff5f72; font-size: {font_size}px; text-align:center;">{text}</p>'
    st.markdown(res, unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title('Air Quality Prediction Project🌩')

st.write(36 * "-")
fancy_header('\n Connecting to Hopsworks Feature Store...')

project = hopsworks.login()

st.write("Successfully connected!✔️")

st.write(36 * "-")
fancy_header('\n Getting data from Feature Store...')

today = date.today()
##########################城市####################
city = "Guangzhou"
df_weather = get_weather_data_weekly(city, today)
df_weather.date = df_weather.date.apply(timestamp_2_time)
df_weather_x = df_weather.drop(columns=["date"]).fillna(0)

########################根据模型名称进行修改#####################
mr = project.get_model_registry()
model = mr.get_model("AIR_Forecast_Model", version=9)
model_dir = model.download()
model = joblib.load(model_dir + "/AIR_Forecast_Model.pkl")

st.write("-" * 36)


preds = model.predict(df_weather_x).astype(int)
pollution_level = get_aplevel(preds.T.reshape(-1, 1))

next_week = [f"{(today + timedelta(days=d)).strftime('%Y-%m-%d')},{(today + timedelta(days=d)).strftime('%A')}" for d in range(8)]

df = pd.DataFrame(data=[preds, pollution_level], index=["AQI", "Air pollution level"], columns=next_week)
###########如果报错AQI这个修改成preds的标签##################

st.write(df)

st.button("Re-run")


