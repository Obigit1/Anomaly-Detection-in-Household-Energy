## Anomaly Detection in Household Energy Consumption

The importance of energy use in household cannot be overemphasised. It has become almost impossible to live without enery use. As part of energy consumption and managment, we looked a the possible ways energy wastage could be detected and prevented using the Anomaly detection system.

In this project, we build an anomaly detection system using both deep and machine learning methods to identify irregularities power usage readings. This system was deployed to streamlit web app for optimal visualisation and control.

Methods

1. Statistical Method presented on a tacble which detected the anomalous behaviour in some of the readings which could imply power wastage during the period.
2. Isolation Forest showed the detected anomalies using the feature, Daily Global Active Power which represents some of the electrical appliances collected by a sensor installed in the meter and the year readings 
3. Autoencoder: This is dimensionality method take the original input data to a lower dimensional and reconstruct it to detect the abnormal values.
4. LSTM played a part in identifying unusual patterns in the data.

The streamlit app (pro_app) displays the results of the abnormalities and other statistical attributes of the data used for this project.
