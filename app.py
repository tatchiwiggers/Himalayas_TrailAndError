import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from joblib import load
import pickle 
import sys



### Making all text white 
st.markdown("""
    <style>
    /* Make all paragraph text white */
    .stApp {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
### Getting a background
import base64

# Function to load and encode image
def load_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# Function to apply image as background
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

# Apply background image
st.markdown(background_image_style("climbing everest.jpg"), unsafe_allow_html=True)
# Add text as title with specific features
st.markdown("""
    <h2 style='text-align: center; color: #FFFFFF; font-family: Georgia;'>
        Welcome to your expedition to the Himalayas! 🏔️
    </h2>
""", unsafe_allow_html=True)

st.markdown("<h3 style='color: white; font-family: Georgia;'>Tell us a bit about yourself and we'll recommend the perfect mountain for you to climb according to your profile.</h3>", unsafe_allow_html=True)

#st.image('climbing everest.jpg', caption="This could be you", use_container_width=True)

## Change color to text input 
st.markdown("""
    <style>
    label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


age = st.text_input("How old are you?")
try:
    age = int(age)
except:
    #st.error("Please make sure that you only enter a number")
    st.stop()


nb_total = st.text_input("How many people will join the expedition?")
try:
    nb_total = int(nb_total)
except:
    st.stop()

nb_hired = st.text_input("How many of those are hired staff?")
try:
    nb_hired = int(nb_hired)
except:
    st.stop()


pct_hired= int(nb_hired)/int(nb_total)
nb_members = int(nb_total) - int(nb_hired)


season_list = ["Spring", "Summer", "Autumn", "Winter"]
with st.container(border=True):
    season = st.selectbox("In which season would you like to hike?", season_list)
    
sex_list = ["M", "F"]
with st.container(border=True):
    sex = st.selectbox("Sex", sex_list)
o2_list = ["Yes", "No"]
with st.container(border=True):
    o2 = st.selectbox("Will you bring oxygen?", o2_list)
if o2 == "Yes":
    o2used = 1
if o2 == "No":
    o2used = 0

difficulty_list = ["Easy","Medium", "Hard", "Extreme" ]
with st.container(border=True):
    difficulty = st.selectbox ("Difficulty level", difficulty_list)
if difficulty == "Easy":
    user_diff_cat = 1
if difficulty == "Medium":
    user_diff_cat = 2
if difficulty == "Hard":
    user_diff_cat = 3
if difficulty == "Extreme":
    user_diff_cat = 4

Country = ['Other',
 'Afghanistan',
 'Albania',
 'Algeria',
 'Andorra',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Bahrain',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Bhutan',
 'Bolivia',
 'Bosnia-Herzegovina',
 'Botswana',
 'Brazil',
 'Bulgaria',
 'Canada',
 'Chile',
 'China',
 'Colombia',
 'Costa Rica',
 'Croatia',
 'Cuba',
 'Cyprus',
 'Czech Republic',
 'Czechoslovakia',
 'Denmark',
 'Dominica',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'El Salvador',
 'Estonia',
 'Finland',
 'France',
 'Georgia',
 'Germany',
 'Greece',
 'Guatemala',
 'Honduras',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Iran',
 'Iraq',
 'Ireland',
 'Israel',
 'Italy',
 'Japan',
 'Jordan',
 'Kazakhstan',
 'Kenya',
 'Kosovo',
 'Kuwait',
 'Kyrgyz Republic',
 'Latvia',
 'Lebanon',
 'Libya',
 'Liechtenstein',
 'Lithuania',
 'Luxembourg',
 'Macedonia',
 'Malaysia',
 'Malta',
 'Mauritius',
 'Mexico',
 'Moldova',
 'Mongolia',
 'Montenegro',
 'Morocco',
 'Myanmar',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Norway',
 'Oman',
 'Pakistan',
 'Palestine',
 'Panama',
 'Paraguay',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Russia',
 'San Marino',
 'Saudi Arabia',
 'Serbia',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sri Lanka',
 'Sweden',
 'Switzerland',
 'Syria',
 'Taiwan',
 'Tajikistan',
 'Tanzania',
 'Thailand',
 'Tunisia',
 'Turkey',
 'UAE',
 'UK',
 'USA',
 'Ukraine',
 'Uruguay',
 'Uzbekistan',
 'Venezuela',
 'Vietnam',
 'Yugoslavia']
   
with st.container(border=True):
    country = st.selectbox("Country", Country)
# Function to get country_max_height  
df= pd.read_csv('Country_max_heigth_list.csv')
def get_highest_peak(country):
    country = country.lower()
    if country in df['Country'].str.lower().values:
        peak = df[df['Country'].str.lower() == country]['Highest_Peak_m']
    else:
        peak = df[df['Country'].str.lower() == 'other']['Highest_Peak_m']
    if not peak.empty:
        return peak.values[0]
country_max_height = get_highest_peak(country)

if st.button("🚀 Confirm and Continue"):
    # Only run this after the button is clicked
    st.write("Thanks! Processing your inputs...")
    # Definition of new data for model 1 
    
    new_data = pd.DataFrame({
        'mseason': [season],
        'sex': [sex],
        'country_max_height': [country_max_height],
        'mo2used': [o2used],
        'nb_members': [nb_members],
        'pct_hired': [pct_hired],
        'age': [age],
    })
    
    # Here we need to run model 1 to get max_height
    
    scaler = load('scaler.joblib')
    encoder = load('encoder (2).joblib')
    model = load('model.joblib')
    
    new_data_num = scaler.transform(new_data.select_dtypes(include="number"))
    new_data_cat = encoder.transform(new_data.select_dtypes(exclude="number"))
    
    new_data_scaled = pd.concat([new_data_num,new_data_cat], axis=1)                                                   
    
    max_height_prediction = int(model.predict(new_data_scaled)[0])
    
    st.write(f"According to our analysis you will be able to climb up to {max_height_prediction} meters!")
    
    # -- Filter -- 
    
    peak_filter = pd.read_csv("peak_filter.csv", index_col=0)
    
    # function to categorize peaks and return a df with a single column "pkname" with the names of the peaks in the user category.
    
    def success_func(user_diff_cat):
        df = peak_filter.copy()
        df['success_cat'] = pd.qcut(
                               df['success_rate'],
                               q=4,
                               labels=[1, 2, 3, 4],
                               duplicates="drop"
                               )
        return df[df['success_cat'] == user_diff_cat][["pkname"]]
    
    # Function to filter the peak list based on the category and the max_height_prediction
    
    def filter(max_height_prediction, user_diff_cat): 
        df = peak_filter.copy()
        filter_output = df[(df.heightm <= max_height_prediction) & (df.pkname.isin(success_func(user_diff_cat)["pkname"]))] \
                    .sort_values(by="nb_members", ascending=False) \
                    .head(3)  \
                     [["peakid", "pkname", "heightm", "death_rate"]]
        return filter_output
    
    # storing output as a variable
    
    filter_output = filter(max_height_prediction, user_diff_cat)
    
    # Dealing with results with less than 3 outputs
    
    #if len(filter_output) != 0:           
    #    filter_output
    #else:
    #    st.write("No peaks match the selected criteria")
    
    #  Definition of new data for model 2
    
    new_data = pd.DataFrame({
        'mseason': [season],
        'sex': [sex],
        'country_max_height': [country_max_height],
        'mo2used': [o2used],
        'nb_members': [nb_members],
        'pct_hired': [pct_hired],
        'age': [age]
    })
    
    # Concating peakid to new_data -> returning it as data_to_model_2
    
    data_to_model_2 = pd.concat([new_data.iloc[[0]]] * 3, ignore_index=True)
    peakid_var = filter_output.reset_index()
    data_to_model_2["peakid"] = peakid_var["peakid"]
    
    #instantiating model2
    scaler2 = load('scaler2.joblib')
    encoder2 = load('encoder2.joblib')
    model2 = load('model2.joblib')
    model2_num = scaler2.transform(data_to_model_2.select_dtypes(include="number"))
    model2_cat = encoder2.transform(data_to_model_2.select_dtypes(exclude="number"))
    data_to_model_2_scaled = pd.concat([model2_num,model2_cat], axis=1)
    success_prob = model2.predict_proba(data_to_model_2_scaled)[:,1]
    success_prob1 = success_prob[0]
    success_prob2 = success_prob[1]
    success_prob3 = success_prob[2]
    st.write(f'{success_prob1}, {success_prob2}, {success_prob3}')
    
    
    st.write(f"According to our analysis you can climb: ")
    
    ### Printing output, we will just need to feed the labels with the right variables 

    label_1 = "death_peak_1"
    main_1 = "name_peak_1"
    note_1 = "success_peak_1"
    note_1_1 = "height_peak_1"
    
    label_2 = "death_peak_2"
    main_2 = "name_peak_2"
    note_2 = "success_peak_2"
    note_2_1 = "height_peak_2"
    
    label_3 = "death_peak_3"
    main_3 = "name_peak_3"
    note_3 = "success_peak_3"
    note_3_1 = "height_peak_3"
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    # First column
    col1.markdown(f"""
        <div style='text-align: center; line-height: 1.2;'>
            <div style='color: red;'>{label_1}</div>
            <div style='color: white; font-size: 28px; font-weight: bold;'>{main_1}</div>
            <div style='color: white;'>{note_1_1}meters</div>
            <div style='color: green; font-size: 22px;'>{note_1}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Second column
    col2.markdown(f"""
        <div style='text-align: center; line-height: 1.2;'>
            <div style='color: red;'>{label_2}</div>
            <div style='color: white; font-size: 28px; font-weight: bold;'>{main_2}</div>
            <div style='color: white;'>{note_2_1}meters</div>
            <div style='color: green; font-size: 22px;'>{note_2}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Third column
    col3.markdown(f"""
        <div style='text-align: center; line-height: 1.2;'>
            <div style='color: red;'>{label_3}</div>
            <div style='color: white; font-size: 28px; font-weight: bold;'>{main_3}</div>
            <div style='color: white;'>{note_3_1}meters</div>
            <div style='color: green; font-size: 22px;'>{note_3}</div>
        </div>
    """, unsafe_allow_html=True)
