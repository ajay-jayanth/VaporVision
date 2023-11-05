from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from streamlit_extras.switch_page_button import switch_page
import joblib

st.sidebar.image('vv.png')

desc = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 45px; font-weight: bold">Input your sensor data</p>'
st.markdown(desc, unsafe_allow_html=True)

desc2 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 18px;">Upload your own sensor record data here and visualize methane patterns over time, as well as predict where leaks will occur!</p>'
st.markdown(desc2, unsafe_allow_html=True)
st.divider()

csv_file = st.file_uploader("Upload your sensor reading csv file: ")

if csv_file is not None:
    df = pd.read_csv(csv_file)
    df.to_csv("user_input.csv")

    max_rows = len(df)
    df2 = df
    st.markdown(
        """
    <style>
    button {
        height: auto;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        margin-left: -5px
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    co1, co2, co3, co4, co5 = st.columns(5)
    # co1.subheader("View Leakage Predictions -> ")
    if (co2.button("View leakage predictions")):
        loaded_model = joblib.load(
            '/Users/yeshn/Desktop/hackutd23/random_forest.pkl')

        # NORMALIZE THE VALUES HERE
        # add code
        # add code

        col_dat = np.array(df[df.columns[2:26]])
        mean = np.mean(col_dat.flatten())
        stdev = np.std(col_dat.flatten())
        col_dat = np.round((col_dat - mean)/stdev, decimals=5)
        print(col_dat)

        # df_predict = df.iloc[:, 2:]
        # print(df_predict)
        y_pred = loaded_model.predict(col_dat)
        print(y_pred)
        # print(df.iloc[:, 1])
        final_df = pd.DataFrame(
            {'time': df.iloc[:, 1], 'leaks': y_pred})
        print(final_df)
        final_df.to_csv('exported.csv', index=False)

        switch_page("Leakage Prediction")
    if (co4.button("View trends on this data")):
        switch_page("Trends")

    st.divider()

    d3 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 28px; font-weight: bold">Visualize your data</p>'
    st.markdown(d3, unsafe_allow_html=True)

    stdevs = {}
    for col in df.columns[2:]:
        stdevs[col] = [df[col].quantile(0.25), df[col].quantile(
            0.5), df[col].quantile(0.75)]

    stdev_df = pd.DataFrame(stdevs)
    stdev_df = stdev_df.transpose()
    stdev_df.columns = ['Lower Quartile', 'Mean', 'Upper Quartile']
    # print(stdev_df)

    stdev_df.index = ['40' + index.split('40', 1)[1]
                      for index in stdev_df.index]
    stdev_df.index = [index.replace('_', ' ', 1).split('_')[
        0] for index in stdev_df.index]

    stdev_df = stdev_df.groupby(stdev_df.index).mean()
    stdev_df = stdev_df.reset_index()
    stdev_df.rename(columns={'index': 'name'}, inplace=True)
    print(stdev_df)

    h, m, s = st.columns(3)
    hr = h.slider("Hours after start", 0, 24, 0, 1)
    min = m.slider("Minutes after start", 0, 60, 0, 1)
    secs = s.slider("Seconds after start", 0, 60, 0, 1)
    time = hr * 3600 + min * 60 + secs

    max_hrs = int(max_rows / 3600)
    max_mins = int((max_rows - (max_hrs * 3600)) / 60)
    max_sec = int((max_rows - (max_hrs * 3600)) % 60)

    # st.write(max_hrs, max_mins, max_sec)
    st.divider()
    if (time > max_rows):
        time = max_rows - 1
        st.text("Viewing map " + str(max_hrs) + " hours, " + str(max_mins) +
                " minutes, " + str(max_sec) + " seconds after start")
    else:
        # st.text("Viewing user data map " + str(hr) + " hours, " + str(min) + " minutes, " +
        #          str(secs) + " seconds after start time: ")
        st.write("Viewing user data map ", hr, " hours, ", min, " minutes, ",
                 secs, " seconds after start time: ")

    IMG_URL = "https://lh4.googleusercontent.com/-GHJ8ii2xewc-8fTFt6w7VjdllPA2ofNn4pBzCMD3W-qXB-8eOEsl0AtwG0lrPgerDP2CazKu1QrhLXMUms2Uqvbw6teNwNil1WhDWwzB6JPQbiYICwwfiKdVl6BFKdc=w1280"

    visited = []
    mapdf = pd.DataFrame(columns=["lat", "long", "methane"])
    for column in range(2, 26, 1):
        name = df.columns[column]
        filter1 = name[7:].strip()
        lat = filter1[:filter1.index("_")]
        filter2 = filter1[filter1.index("_")+1:]
        long = filter2[:filter2.index("_")]
        coordinate = lat + " " + long

        print(lat, long)
        if coordinate not in visited:
            selected_row = stdev_df[stdev_df['name'] == coordinate]
            colors = ""
            trueML = float(df.iloc[time, column])
            if (trueML < selected_row["Mean"].values[0]):
                colors = [0, 250, 0, 160]
            elif (trueML < selected_row["Upper Quartile"].values[0]):
                colors = [255, 204, 0, 160]
            else:
                colors = [220, 30, 0, 160]
            print(colors)

            methaneLev = float(df.iloc[time, column]) * \
                float(df.iloc[time, column]) * 2 / 1000000
            mapdf = mapdf.append(
                {'lat': float(lat), 'long': float(long), "methane": float(methaneLev), "color": colors}, ignore_index=True)

            visited.append(coordinate)

    BOUNDS = [
        [-105.140900, 40.595350],
        [-105.140900, 40.596200],
        [-105.138851, 40.596200],
        [-105.138851, 40.595350]
    ]

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
                    get_color="color",
                    get_radius="[methane]",
                ),
            ],
        )
    )

    c1, c2, c3 = st.columns(3)
    c1.write(":red_circle: - High methane levels")
    c2.write(":large_yellow_circle: - Moderate methane levels")
    c3.write(":large_green_circle: - Low methane levels")
    st.divider()
