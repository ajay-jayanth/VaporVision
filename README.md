# VaporVision
![Dark Blue with Space Photos Movie Ticket-5](https://github.com/Sarvesh-Sathish/react-website-v1/assets/34663815/bcc57bd5-3646-4f4f-a114-3ce03f817bb5)
## Inspiration
EOG generously provided us with key data from their 24 methane sensors which enabled us to work towards a problem that helps the environment by catching methane leaks. This dataset inspired the development of a comprehensive web application that leverages data visualization and AI/ML techniques such as random forest generators. EOG's engineering team has recognized the need for a sophisticated tool to assess and visualize data from the methane sensors. In response, we have proposed the creation of a full-stack application that offers a range of valuable features to support EOG's efforts.

## What it does
<img width="1180" alt="Screen Shot 2023-11-05 at 9 39 41 AM" src="https://github.com/Sarvesh-Sathish/react-website-v1/assets/34663815/3c8b0eed-e657-43be-bff7-3f50b427c298">
### Methane Sensor Trend Modeling/Mapping
Based on the time, we display general labels of each sensor on the map based on the methane levels detected at each sensor. We do this for both the training and new data. There is also a trend graph of each sensor allowing the user to see when the methane spiked on the graph.

### Methane Leak Anomaly Detection
EOG provided us with data collected from methane sensors placed at strategic locations around a certain facility. We used an advanced anomaly detection model to determine the regions where the gas was spread to by allowing the model to determine which leak triggered which sensor. This was then tested against a cross-validation set split which yielded an 88% accuracy.

![edde7494-2116-470b-a745-2b2277a448e9](https://github.com/ajay-jayanth/OptiMine/assets/69321866/ef9c7446-93db-4268-a452-e7aae7c28130)

### Gaussian Plume Modelling
We computed the Gaussian plume model equation to obtain the concentration values over a set of 2D longitudinal and latitude coordinates on the map. Once that was computed, a threshold was set for a certain range of values and those densities set up the heat map for specific methane leaks.
![2_eq](https://github.com/Sarvesh-Sathish/react-website-v1/assets/34663815/4f76907a-ecb2-4add-b886-460cd0ad7253)

The gradient coefficients were computed based on the type of weather that occurred on that day. There were 5 sets of weather coefficients based on the barometric pressure & weather data which spanned specific ranges. The gradient coefficients allowed us to identify the concentration spread amongst the downwind distance of the source whereas the crosswind distance was dependent on the y value.

## How we built it
### UI/UX Design

We utilized Figma for the layout and design of our user interface, which allowed us to conceptualize the coding aspect with a clear visual representation. Through Figma, we could effectively organize and arrange the user interface components, thus enabling us to anticipate the appearance of our final application. For the front-end development, our predominant choice was Streamlit. Streamlit is a Python library that facilitates interactive data engagement, including manipulation, visualization, and analysis with popular Python tools such as Pandas, Matplotlib, and Scikit-learn. Additionally, Streamlit supports seamless integration with scripting languages, namely HTML and CSS, which were instrumental in the development of certain elements within our web application. 

### Backend Development
####Data Preprocessing

At first glance, the sensor readings were very biased and inconsistent with each other, which if left untreated, would lead to biased data analysis and anomaly detection. To combat this, we preprocessed the data with standardization, allowing for the data in each sensor reading to be consistent with each other within a normal distribution. This allowed for fluent data analysis and unbiased predictions.

#### Data Computation

To provide the anomaly detection model with its training data, the given data needed to be formatted and aggregated to the requirements of the training procedure. First, the sensor readings were compiled into 50-second timeframes by aggregating the sensor readings by mean in those intervals. Then, each time frame was tagged with the region or regions where a leak occurred so that the model could The resulting data would allow the model to detect anomalies in methane levels and associate them with the leak events that occurred within the timeframes.


## Challenges we ran into
### Overfitting of the machine learning model
Because the data was so similar across the 24 hour time frame in the dataset, testing and validating the model was a challenge. At first we were excited to be getting near 1.0 accuracy, but we soon realized that there was overfitting happening. We tried many approaches, but ended up using scikit-learn's random forest generator classifier model. To effectively validate the model with data that was so similar, we had to split the data in by the time rather than by random. This allowed us to effectively validate our model and tune it accordingly.

### Gaussian Plume Function
Our team encountered many challenges in developing the Gaussian Plume Modeling algorithm that modeled how the concentration spread of methane was affected by wind conditions. First, we attempted to naively plug in values into random formulas found online, but they were either too simple or too generic for our problem at hand. After many failed attempts, we dissected the algorithm’s components and through countless iterations, we analyzed its intricacies and successfully implemented a Gaussian Plume Modeling system from the ground up.

###Map API in Streamlit
Creating a map that we could draw sensor and leakage data on proved to be a big challenge. Many maps were not up to data and did not show the facilities near Fort Collins. Some of the maps did not allow us to draw anything on them. Eventually, we found mapbox and pydeck which allowed us to visualize the facilities and draw the necessary data on it.

## Accomplishments that we're proud of

We were extremely proud of our plume model that took over 10 hours to complete. The algorithm had a lot of parameters and test cases that we had to account for which was not on the original article link in the github. We had to research the various weather condition cases and how the plum cloud would spread when emitted through a gas leak. There were not a lot of equations online that had consistent constants because of a change of units amongst each website. However, we researched deep into the topic and eventually were able to derive the constants ourselves (‘a’, ‘b’, ‘c’, ‘d’, ‘f’) using pressure and wind speed on that particular time interval.

## What we learned

Our team learned how to effectively communicate under an immense load of pressure as well as how to approach a daunting project one step at a time. We gained skills in data analysis using the Pandas and Numpy frameworks. 

## What's next for VaporVision

In upcoming scenarios, EOG Resources can utilize their sensor data, this enhanced software can be harnessed to detect methane leaks in real-time. The collected sensor data serves as a valuable resource for pinpointing and addressing potential leaks, ensuring the safety and integrity of their facilities.

Furthermore, this data optimization tool can be instrumental in evaluating the efficiency of EOG's latest sensor technology for methane leak detection, comparing their performance with previous sensor models used in prior expeditions. By doing so, it allows EOG to continually improve and enhance their leak detection capabilities for safer and more effective facilities.

## To Run
streamlit run Home.py

