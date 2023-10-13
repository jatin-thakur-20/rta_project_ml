import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier

from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/ExtraTreesClassifier.joblib')

st.set_page_config(page_title='Accident Severity Prediction',page_icon='🚨',layout='centered', initial_sidebar_state='auto')


st.title("Accident Severity Prediction 🚨")

options_casualties = [1, 2, 3, 4, 5, 6, 7, 8]
options_light_conditions = ['Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting', 'Daylight']
options_vehicles_involved = [1, 2, 3, 4, 6, 7]
options_age = ['18-30', '31-50', 'Over 51', 'Under 18', 'Unknown']
options_day = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
options_road_surface = ['Dry', 'Flood over 3cm. deep', 'Snow', 'Wet or damp']
options_driving_experience = ['1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'Below 1yr', 'No Licence', 'unknown']
options_Junction = ['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'Unknown', 'X Shape', 'Y Shape']
options_hour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
#options_lanes  =['Double carriageway (median)', 'One way', 'Two-way (divided with broken lines road marking)', 'Two-way (divided with solid lines road marking)', 'Undivided Two way', 'Unknown', 'other']
options_vehicle_type = ['Automobile','Bajaj','Bicycle', 'Long lorry', 'Lorry (11?40Q)', 'Lorry (41?100Q)', 'Motorcycle', 'Other', 'Pick up upto 10Q', 'Public (12 seats)', 'Public (13?45 seats)', 'Public (> 45 seats)', 'Ridden horse', 'Special vehicle', 'Stationwagen', 'Taxi', 'Turbo']

dictionary = {0: 'Fatal injury', 1: 'Serious Injury', 2: 'Slight Injury'}


with st.form('prediction_form'):
    st.subheader("Features for prediction")

    hour = st.select_slider("Hour of accident",options_hour)
    day = st.selectbox("Day of week", options_day)
    age = st.selectbox("Driver Age",options_age)
    driver_exp = st.selectbox("Driver Experience",options_driving_experience)
    vehicles = st.select_slider("Number of Vehicles involved",options_vehicles_involved)
    casualties = st.select_slider("Number of casualties",options_casualties)
    light_conditions = st.selectbox("Light Conditions",options_light_conditions)
    road = st.selectbox("Road Surface Conditions",options_road_surface)
    junction = st.selectbox("Road Junction Type",options_Junction)
    vehicle_type = st.selectbox("Type of Vehicle",options_vehicle_type)


    submit = st.form_submit_button("Prediction!")

    if submit:
        hour_ = ordinal_encoder(hour,options_hour)
        day_ = ordinal_encoder(day,options_day)
        age_ = ordinal_encoder(age,options_age)
        driver_exp_ = ordinal_encoder(driver_exp,options_driving_experience)
        vehicles_ = ordinal_encoder(vehicles,options_vehicles_involved)
        casualties_ = ordinal_encoder(casualties,options_casualties)
        light_conditions_ = ordinal_encoder(light_conditions,options_light_conditions)
        road_ = ordinal_encoder(road, options_road_surface)
        junction_ = ordinal_encoder(junction,options_Junction)
        #lanes_ = ordinal_encoder(lanes, options_lanes)
        vehicle_type_ = ordinal_encoder(vehicle_type,options_vehicle_type)

        data = np.array([light_conditions_,vehicles_,casualties_,age_,day_,driver_exp_,road_,junction_,hour_,vehicle_type_]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"Accident Severity Prediction: {dictionary[pred[0]]}")