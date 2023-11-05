from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages, add_page_title

st.sidebar.image('vv.png')

original_title = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 50px;">Where are there leaks?</p>'
st.markdown(original_title, unsafe_allow_html=True)

desc2 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 18px;">A personalized ML model to identify where there may be methane leaks based on the sensor data provided by you!</p>'
st.markdown(desc2, unsafe_allow_html=True)

IMG_URL = "https://lh4.googleusercontent.com/-GHJ8ii2xewc-8fTFt6w7VjdllPA2ofNn4pBzCMD3W-qXB-8eOEsl0AtwG0lrPgerDP2CazKu1QrhLXMUms2Uqvbw6teNwNil1WhDWwzB6JPQbiYICwwfiKdVl6BFKdc=w1280"
COLOR_URL = "https://htmlcolorcodes.com/assets/images/colors/dark-blue-color-solid-background-1920x1080.png"

st.divider()

df = pd.read_csv("exported.csv")

max_len = len(df)
time = st.slider("Enter time since start for leakage prediction: ", 0, max_len)

st.divider()

time_in_df = df.iloc[time, df.columns.get_loc("time")]
# st.write("This time is marked as ", time_in_df, " seconds in the dataframe.")

areas = df.iloc[time, df.columns.get_loc("leaks")]
areas = f"{areas:05}"

count = 0
for a in areas:
    if a == '1':
        count += 1

# if (count == 1):
#     st.write("At time ", time, " seconds after start (marked as ",
#              time_in_df, " seconds), there is  `1` region with a leak.")
# else:
#     st.write("At time ", time, " seconds after start (marked as ",
#              time_in_df, " seconds), there are", count, "regions with a leak.")


BOUNDS = [
    [-105.140900, 40.595350],
    [-105.140900, 40.596200],
    [-105.138851, 40.596200],
    [-105.138851, 40.595350]
]

mapdf = pd.DataFrame(columns=["lat", "long"])

max_hrs = int(time / 3600)
max_mins = int((time - (max_hrs * 3600)) / 60)
max_sec = int((time - (max_hrs * 3600)) % 60)

st.write("At ", max_hrs, " hours, ", max_mins,
         " minutes, and ", max_sec, " seconds after start: ")
# d3 = '<p style="font-family: Courier New; color:white; text-align: center; font-size: 28px; font-weight: bold">Methane Level</p>'
# st.markdown(d3, unsafe_allow_html=True)

o1, o2, o3 = st.columns(3)

if (areas == "00000"):
    o1.write("\t\t - There is no predicted leak!")

if (areas != "0"):
    if (areas[0] != "0"):
        mapdf = mapdf.append(
            {'lat': 40.595807, 'long': -105.139850}, ignore_index=True)
        o1.write("\t\t - Predicted leak at `4T`")
    if (areas[1] != "0"):
        mapdf = mapdf.append(
            {'lat': 40.595927, 'long': -105.139402}, ignore_index=True)
        o1.write("\t\t - Predicted leak at `5S`")
    if (areas[2] != "0"):
        mapdf = mapdf.append(
            {'lat': 40.595659, 'long': -105.139425}, ignore_index=True)
        o1.write("\t\t - Predicted leak at `5W`")
    if (areas[3] != "0"):
        mapdf = mapdf.append(
            {'lat': 40.595948, 'long': -105.140328}, ignore_index=True)
        o1.write("\t\t - Predicted leak at `4W`")
    if (areas[4] != "0"):
        mapdf = mapdf.append(
            {'lat': 40.595648, 'long': -105.140300}, ignore_index=True)
        o1.write("\t\t - Predicted leak at `4S`")

if (o3.button("Download output .txt file")):
    data = df
    # Function to convert leaks to locations

    def convert_leaks_to_location(leaks):
        if leaks == 0:
            return "None"

        locations = []
        labels = ['4T', '5S', '5W', '4W', '4S']
        leaks_str = str(leaks).zfill(5)

        for i, label in enumerate(labels):
            if leaks_str[i] == '1':
                locations.append(label)

        return "|".join(locations)

    # Apply the function to the DataFrame
    data['location'] = data['leaks'].apply(convert_leaks_to_location)

    # Create a list of formatted lines
    formatted_lines = []
    for _, row in data.iterrows():
        formatted_lines.append(f"{int(row['time'])}, {row['location']}")

    # Write the lines to a text file
    with open("VaporVision.txt", "w") as file:
        file.write("\n".join(formatted_lines))


st.pydeck_chart(
    pdk.Deck(
        # 'light', 'dark', 'satellite', 'road'
        map_style='mapbox://styles/mapbox/satellite-streets-v12',
        initial_view_state=pdk.ViewState(
            latitude=40.595790,
            longitude=-105.139859,
            zoom=18,
            pitch=1,
        ),
        layers=[
            pdk.Layer(
                "BitmapLayer",
                data=None,
                image=IMG_URL,
                bounds=BOUNDS,
                opacity=0.5
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=mapdf,
                get_position="[long, lat]",
                get_color="[139,14,236]",
                get_radius="15",
                opacity=0.5
            ),
        ],

    )
)
