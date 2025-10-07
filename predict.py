import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load all files

with open("model_files/model_lin_reg.pkl", "rb") as file_1:
    model_lin_reg = pickle.load(file_1)

with open("model_files/model_scaler.pkl", "rb") as file_2:
    model_scaler = pickle.load(file_2)

with open("model_files/model_encoder.pkl", "rb") as file_3:
    model_encoder = pickle.load(file_3)

with open("model_files/list_num_cols.txt", "r") as file_4:
    list_num_cols = json.load(file_4)

with open("model_files/list_cat_cols.txt", "r") as file_5:
    list_cat_cols = json.load(file_5)


def run():
    # judul
    st.title("Predict Player Rating")
    # user input
    with st.form(key="player"):
        # input nama
        st.header("Masukan data pemain")

        name = st.text_input("Masukan anama pemain", placeholder="cth: Rajib Kurniawan")
        age = st.number_input("Masukan usia pemain", min_value=0, max_value=100, value=20)
        height = st.number_input(
            "Masukan tinggi pemain",
            min_value=0,
            max_value=300,
            value=170,
            help="Tinggi dalam cm",
        )
        weight = st.number_input(
            "Masukan berat pemain",
            min_value=0,
            max_value=100,
            value=6,
            help="Berat dalam kg",
        )
        price = st.number_input(
            "Masukan tinggi pemain", min_value=0, value=500000, help="Harga dalam uero"
        )
        st.write("___")

        # workrate
        attcaking_wr = st.selectbox("Attacking work rate", ["Low", "Medium", "High"])
        deffendsive_wr = st.selectbox("Deffendsive work rate", ["Low", "Medium", "High"])
        st.write("___")

        # total columns
        pace = st.slider("Pace total", min_value=0, max_value=100, value=50)
        shooting = st.slider("shooting total", min_value=0, max_value=100, value=50)
        passing = st.slider("passing total", min_value=0, max_value=100, value=50)
        driblling = st.slider("dribilling total", min_value=0, max_value=100, value=50)
        defending = st.slider("defending total", min_value=0, max_value=100, value=50)
        pyshicality = st.slider("pyshicality total", min_value=0, max_value=100, value=50)
        st.write("___")

        # submit button
        submit = st.form_submit_button("predict")

    if submit:
            data_inf = {
                "Name": name,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "Price": price,
                "AttackingWorkRate": attcaking_wr,
                "DefensiveWorkRate": deffendsive_wr,
                "PaceTotal": pace,
                "ShootingTotal": shooting,
                "PassingTotal": passing,
                "DribblingTotal": driblling,
                "DefendingTotal": defending,
                "PhysicalityTotal": pyshicality,
            }

            data_inf = pd.DataFrame([data_inf])
            st.dataframe(data_inf)

            # processing data
            data_inf_num = data_inf[list_num_cols]
            data_inf_cat = data_inf[list_cat_cols]

            ## Feature Scaling
            data_inf_num_scaled = model_scaler.transform(data_inf_num)

            ## Feature Encoding
            data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

            ## Concate
            data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

            # Predict using Linear Regression
            y_pred_inf = model_lin_reg.predict(data_inf_final)
            st.write("# Prediction", int(y_pred_inf[0]))

if __name__ == "__main__":
    run()
