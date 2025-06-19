import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os

# === CACHING UTILITIES ===
@st.cache_resource
def load_model_components():
    scaler = joblib.load('scaler.joblib')
    encoder = joblib.load('encoder (2).joblib')
    model = joblib.load('model.joblib')
    return scaler, encoder, model

@st.cache_resource
def load_model2_components():
    scaler2 = joblib.load('scaler2.joblib')
    encoder2 = joblib.load('encoder2.joblib')
    model2 = joblib.load('model2.joblib')
    return scaler2, encoder2, model2

@st.cache_data
def load_country_data():
    return pd.read_csv('Country_max_heigth_list.csv')

@st.cache_data
def load_peak_data():
    return pd.read_csv("peak_filter.csv", index_col=0)

# === BACKGROUND IMAGE ===
def load_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

def background_image_style(path):
    encoded = load_image(path)
    return f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                          url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

st.markdown(background_image_style("climbing everest.jpg"), unsafe_allow_html=True)

# === TITLE ===
st.markdown("""
    <h2 style='text-align: center; color: #FFFFFF; font-family: Georgia;'>
        Welcome to your expedition to the Himalayas! 🌄️
    </h2>
""", unsafe_allow_html=True)

st.markdown("<h3 style='color: white; font-family: Georgia;'>Tell us a bit about yourself and we'll recommend the perfect mountain for you to climb.</h3>", unsafe_allow_html=True)

# === INPUTS ===
st.markdown("""
    <style>
    label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

age = st.text_input("How old are you?")
if not age.isdigit():
    st.stop()
age = int(age)

nb_total = st.text_input("How many people will join the expedition?")
if not nb_total.isdigit():
    st.stop()
nb_total = int(nb_total)

nb_hired = st.text_input("How many of those are hired staff?")
if not nb_hired.isdigit():
    st.stop()
nb_hired = int(nb_hired)

pct_hired = nb_hired / nb_total
nb_members = nb_total - nb_hired

season = st.selectbox("In which season would you like to hike?", ["Spring", "Summer", "Autumn", "Winter"])
sex = st.selectbox("Sex", ["M", "F"])
o2 = st.selectbox("Will you bring oxygen?", ["Yes", "No"])
o2used = 1 if o2 == "Yes" else 0
difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard", "Extreme"])
diff_map = {"Easy": 1, "Medium": 2, "Hard": 3, "Extreme": 4}
user_diff_cat = diff_map[difficulty]

country = st.selectbox("Country", load_country_data()['Country'].unique().tolist())

def get_highest_peak(country, df):
    match = df[df['Country'].str.lower() == country.lower()]
    if not match.empty:
        return match['Highest_Peak_m'].values[0]
    return df[df['Country'].str.lower() == 'other']['Highest_Peak_m'].values[0]

country_max_height = get_highest_peak(country, load_country_data())

if st.button("🚀 Confirm and Continue"):
    st.write("Thanks! Processing your inputs...")

    new_data = pd.DataFrame({
        'mseason': [season],
        'sex': [sex],
        'country_max_height': [country_max_height],
        'mo2used': [o2used],
        'nb_members': [nb_members],
        'pct_hired': [pct_hired],
        'age': [age],
    })

    scaler, encoder, model = load_model_components()
    num = scaler.transform(new_data.select_dtypes(include="number"))
    cat = encoder.transform(new_data.select_dtypes(exclude="number"))
    new_data_scaled = pd.concat([num, cat], axis=1)
    max_height_prediction = int(model.predict(new_data_scaled)[0])

    st.write(f"According to our analysis, you will be able to climb up to {max_height_prediction} meters!")

    peak_filter = load_peak_data()

    peak_filter['success_cat'] = pd.qcut(
        peak_filter['success_rate'],
        q=4,
        labels=[1, 2, 3, 4],
        duplicates="drop"
    )

    filtered = peak_filter[(peak_filter.heightm <= max_height_prediction) &
                           (peak_filter.success_cat == user_diff_cat)]
    filtered = filtered.sort_values(by="nb_members", ascending=False).head(3)

    base_row = new_data.iloc[0].to_dict()
    data_to_model_2 = pd.DataFrame([base_row for _ in range(len(filtered))])
    data_to_model_2["peakid"] = filtered.reset_index()["peakid"]

    scaler2, encoder2, model2 = load_model2_components()
    num2 = scaler2.transform(data_to_model_2.select_dtypes(include="number"))
    cat2 = encoder2.transform(data_to_model_2.select_dtypes(exclude="number"))
    final_input = pd.concat([num2, cat2], axis=1)
    success_prob = model2.predict_proba(final_input)[:, 1]

    st.write("According to our analysis, you can climb:")

    cols = st.columns(len(filtered))
    for i, row in filtered.iterrows():
        idx = list(filtered.index).index(i)
        with cols[idx]:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='color: red;'>Death Rate: {row['death_rate']:.1f}%</div>
                    <div style='color: white; font-size: 28px; font-weight: bold;'>{row['pkname']}</div>
                    <div style='color: white;'>{row['heightm']} meters</div>
                    <div style='color: green; font-size: 22px;'>Success: {success_prob[idx]:.0%}</div>
                </div>
            """, unsafe_allow_html=True)
