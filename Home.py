import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import matplotlib.image as mimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages, add_page_title

st.sidebar.image('vv.png')

title = '<p style="font-family: Sans Serif; color:white; text-align: center; font-size: 30px;">FAQ</p>'
st.sidebar.markdown(title, unsafe_allow_html=True)

with st.sidebar.expander("What is VaporVision?", expanded=False):
    st.write("VaporVision is a web app that allows you to utilize a `personalized machine learning model` to predict whether methane levels in your area are at risk of a leak.")

with st.sidebar.expander("How do I use this program?", expanded=False):
    st.write("1. Look through the visual analysis of the training data to see what the ML model has been trained on.")
    st.write(
        "2. Navigate to the \'View your data\' tab to input your own sensor recordings.")
    st.write(
        "3. You can visualize your sensor data, view trends, or get leakage prediction insights!")

with st.sidebar.expander("What ML model is being used?", expanded=True):
    st.write("We are utilizing a `RandomForestClassifier` to identify whether your methane levels in a certain area are leaking.")

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be
show_pages(
    [
        Page("Home.py", "Home", "🏠"),
        Page("pages/Input.py", "View your data", ":bar_chart:"),
        Page("pages/Leakage.py", "Leakage Prediction", ":dash:"),
        # Page("pages/Chat.py", "Chat with AI!", ":left_speech_bubble:"),
        Page("pages/Trends.py", "Trends", ":up:")
    ]
)

# title
original_title = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 100px; font-weight: normal">VaporVision</p>'
st.markdown(original_title, unsafe_allow_html=True)

desc = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 18px;">View the ML model\'s training data logistics here!</p>'
st.markdown(desc, unsafe_allow_html=True)

desc2 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 18px;">To input your own data, go to \'View your data\'</p>'
st.markdown(desc2, unsafe_allow_html=True)

csv_file = "sensor_readings.csv"
st.divider()

h, m, s = st.columns(3)
hr = h.slider("Hours after start", 0, 24, 0, 1)
min = m.slider("Minutes after start", 0, 60, 0, 1)
secs = s.slider("Seconds after start", 0, 60, 0, 1)
# time = st.slider(
#     "Slide to time after start (in seconds) you wish to view methane levels (max 24 hours)", 0, 83400, 0, 100)
time = hr * 3600 + min * 60 + secs

st.divider()
d3 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 28px; font-weight: bold">Methane Level Map</p>'
st.markdown(d3, unsafe_allow_html=True)
# hrs = int(time/3600)
# mins = int((time - (hrs * 3600)) / 60)
# sec = int((time - (hrs * 3600)) % 60)
if (time > 83400):
    time = 83400
    st.text("Viewing training data map 24 hours after start time: ")
else:
    # st.text("Viewing training data map " + str(hr) + " hours, " + str(min) + " minutes, " +
    #          str(secs) + " seconds after start time: ")
    st.write("Viewing training data map ", hr, " hours, ", min, " minutes, ",
             secs, " seconds after start time: ")


df = pd.read_csv(csv_file)

IMG_URL = "https://lh4.googleusercontent.com/-GHJ8ii2xewc-8fTFt6w7VjdllPA2ofNn4pBzCMD3W-qXB-8eOEsl0AtwG0lrPgerDP2CazKu1QrhLXMUms2Uqvbw6teNwNil1WhDWwzB6JPQbiYICwwfiKdVl6BFKdc=w1280"

means = pd.read_csv("stdev.csv")
means = means.groupby('name').mean().reset_index()
print(means)
visited = []

mapdf = pd.DataFrame(columns=["lat", "long", "methane"])
for column in range(2, 26, 1):
    name = df.columns[column]
    filter1 = name[7:].strip()
    lat = filter1[:filter1.index("_")]
    filter2 = filter1[filter1.index("_")+1:]
    long = filter2[:filter2.index("_")]
    coordinate = lat + " " + long

    if coordinate not in visited:
        selected_row = means[means['name'] == coordinate]
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
d4 = '<p style="font-family: Trebuchet MS; color:white; text-align: center; font-size: 28px; font-weight: bold">Gaussian-Plume Modeling System</p>'
st.markdown(d4, unsafe_allow_html=True)

df = pd.read_csv('gaussian_plume.csv')
# val = st.slider("Enter time in seconds", df['time'].iloc[0]-df['time'].iloc[0], df['time'].iloc[-1]-df['time'].iloc[0], 10000.)
val = time
idx = int(val // 50)


class receptorGrid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(y, z, x)


class pointSource:
    def __init__(self, x, y, z, rate, H):
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate
        self.H = H
        self.sourceType = 'point'


class areaSource:
    def __init__(self, x0, dx, nx, y0, dy, ny, z, rate, H):
        self.x = np.linspace(x0, nx*dx, nx+1)
        self.y = np.linspace(y0, ny*dy, ny+1)
        self.z = z
        self.sourceType = 'area'
        self.yMesh, self.zMesh, self.xMesh = np.meshgrid(self.y, z, self.x)
        self.H = H
        self.rate = rate
        self.dx = dx
        self.dy = dy


class stabilityClass:
    def __init__(self, letter):
        self.letter = letter

        if letter == 'A':
            Iy = -1.104
            Jy = 0.9878
            Ky = -0.0076

            Iz = 4.679
            Jz = -1.7172
            Kz = 0.2770

        elif letter == 'B':
            Iy = -1.634
            Jy = 1.0350
            Ky = -0.0096

            Iz = -1.999
            Jz = 0.8752
            Kz = 0.0136

        elif letter == 'C':
            Iy = -2.054
            Jy = 1.0231
            Ky = -0.0076

            Iz = -2.341
            Jz = 0.9477
            Kz = -0.0020

        elif letter == 'D':
            Iy = -2.555
            Jy = 1.0423
            Ky = -0.0087

            Iz = -3.186
            Jz = 1.1737
            Kz = -0.0316

        elif letter == 'E':
            Iy = -2.754
            Jy = 1.0106
            Ky = -0.0064

            Iz = -3.783
            Jz = 1.3010
            Kz = -0.0450

        elif letter == 'F':
            Iy = -3.143
            Jy = 1.0148
            Ky = -0.0070

            Iz = -4.490
            Jz = 1.4024
            Kz = -0.0540

        def sy(dist):
            return np.exp(Iy + Jy*np.log(dist) + Ky*(np.log(dist)**2))

        def sz(dist):
            return np.exp(Iz + Jz*np.log(dist) + Kz*(np.log(dist)**2))

        self.sz = sz
        self.sy = sy


class gaussianPlume:
    def __init__(self, source, grid, stability, U):
        self.grid = grid
        self.source = source
        self.stability = stability
        self.U = U

    def calculateConcentration(self):
        conc = np.zeros_like(self.grid.xMesh, dtype=float)

        if self.source.sourceType == 'area':
            for x in self.source.x:
                for y in self.source.y:
                    a = self.source.rate*self.source.dx*self.source.dy / \
                        (2 * np.pi * self.U * self.stability.sy(self.grid.xMesh - x)
                         * self.stability.sz(self.grid.xMesh - x))
                    b = np.exp(-(self.grid.yMesh - y)**2 /
                               (2*self.stability.sy(self.grid.xMesh - x)**2))
                    c = np.exp(-(self.grid.zMesh-self.source.H)**2/(2*self.stability.sz(self.grid.xMesh - x)**2)) + \
                        np.exp(-(self.grid.zMesh+self.source.H)**2 /
                               (2*self.stability.sz(self.grid.xMesh - x)**2))
                    conc += a*b*c

        if self.source.sourceType == 'point':
            x = self.source.x
            y = self.source.y
            a = self.source.rate / (2 * np.pi * self.U * self.stability.sy(
                self.grid.xMesh - x) * self.stability.sz(self.grid.xMesh - x))
            b = np.exp(-(self.grid.yMesh - y)**2 /
                       (2*self.stability.sy(self.grid.xMesh - x)**2))
            c = np.exp(-(self.grid.zMesh-self.source.H)**2/(2*self.stability.sz(self.grid.xMesh - x)**2)) + \
                np.exp(-(self.grid.zMesh+self.source.H)**2 /
                       (2*self.stability.sz(self.grid.xMesh - x)**2))
            conc += a*b*c

        return conc


def plt_plume(regions, direction):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for region in regions:

        rate = 1.  # g/s/m2
        H = 0.  # m
        U = 5.  # m/s
        xGrid = np.linspace(0, 2228, 100)  # m
        yGrid = np.linspace(0, 1164, 100)  # m
        zGrid = 10.  # m

        if region == '4W':
            if direction == 0:
                pos = [510, 400]
            else:
                pos = [1400, 350]
        elif region == '4T':
            if direction == 0:
                pos = [1150, 500]
            else:
                pos = [950, 600]
        elif region == '4S':
            if direction == 0:
                pos = [700, 800]
            else:
                pos = [600, 400]
        elif region == '5S':
            if direction == 0:
                pos = [1500, 800]
            else:
                pos = [750, 800]
        elif region == '5T':
            if direction == 0:
                pos = [1500, 400]
            else:
                pos = [1400, 850]
        PS = pointSource(pos[0], pos[1], 0, rate, H)
        grid = receptorGrid(xGrid, yGrid, zGrid)
        stability = stabilityClass('C')

        a = gaussianPlume(PS, grid, stability, U)

        concField = a.calculateConcentration()
        concField = concField[0]
        if direction == 1:
            concField = np.rot90(concField, k=1)

        original_cmap = plt.cm.YlOrRd
        colors = original_cmap(np.linspace(0, 1, 256))
        colors[int(256*0.07):, 3] = colors[int(256*0.07):, 3] * 0.2
        colors[:int(256*0.07), 3] = 0

        new_cmap = LinearSegmentedColormap.from_list("ModifiedCmap", colors)
        c = ax.contourf(grid.xMesh[0], grid.yMesh[0],
                        concField, 100, cmap=new_cmap)

    image = plt.imread('map2.jpg')
    ax.imshow(image)
    ax.axis('off')
    plt.tight_layout()
    return fig


def call_plume(df_row):
    cols = ['4T', '5S', '5W', '4W', '4S']
    present = []
    for col in cols:
        if df_row[col] == 1:
            present.append(col)
    if len(present) > 2:
        present = present[:2]
    fig = plt_plume(present, df_row['direction'])
    st.pyplot(fig)


call_plume(df.iloc[idx])
