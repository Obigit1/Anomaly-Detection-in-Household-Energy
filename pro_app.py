import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('household_power_consumption.txt', sep=';', 
                       parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True, 
                       low_memory=False, na_values=['nan', '?'])
    data.fillna(method='ffill', inplace=True)
    cols_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)
    data.set_index('Datetime', inplace=True)
    daily_data = data.resample('D').sum()
    daily_data.dropna(inplace=True)
    return daily_data

# Load the dataset
daily_data = load_data()

# Sidebar for feature selection
st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Select from the List', 
                              ['Home Page', 'Daily Data', 'EDA', 'Correlation Matrix', 'Statistical Anomalies', 
                               'Isolation Forest Anomalies', 'Autoencoder Anomalies', 
                               'LSTM Anomalies'])

# Home page
if option == "Home Page":
        st.markdown("##  Dissertation (COM726) ")
        st.markdown("### Anomaly Detection in Household Energy Consumption Patterns using Deeep Learning ")
        st.markdown("##### Obinna G. Ugwuegbu (Q102104484)")
        st.markdown("##### Supervisor: Dr. Zuhaib Khan ")
        st.write("###### Use the navigation bar on the left to explore the content")
# Daily Data
elif option == 'Daily Data':
    st.write("Daily Aggregated Data:")
    st.dataframe(daily_data)

# EDA - Exploratory Data Analysis
elif option == 'EDA':
    st.write("Exploratory Data Analysis")
    feature = st.selectbox('Select feature to display:', daily_data.columns)

    # Plot time series of selected feature
    st.write(f"Time Series Plot of {feature}")
    st.line_chart(daily_data[feature])

    # Plot histogram of selected feature
    st.write(f"Histogram of {feature}")
    fig, ax = plt.subplots(figsize=(10, 5))
    daily_data[feature].hist(bins=50, ax=ax)
    ax.set_title(f'Histogram of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


    # Plot Rolling Mean and Standard Deviation of selected feature
    st.write(f"Rolling Mean and Standard Deviation of {feature}")

    # Calculate rolling mean and standard deviation
    rolling_mean = daily_data[feature].rolling(window=30).mean()
    rolling_std = daily_data[feature].rolling(window=30).std()

    # Plot rolling statistics
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(daily_data[feature], label=feature)
    ax.plot(rolling_mean, label='Rolling Mean', color='red')
    ax.plot(rolling_std, label='Rolling Std Dev', color='black')
    ax.legend()
    ax.set_title(f'Rolling Mean and Standard Deviation of {feature}')
    st.pyplot(fig)

    # Plot Box Plot of selected feature
    st.write(f"Box Plot of {feature}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=daily_data[feature], ax=ax)
    ax.set_title(f'Box Plot of {feature}')
    st.pyplot(fig)

    # Seasonal Decomposition
    st.write("Seasonal Decomposition")
    if st.button('Decompose'):
        result = seasonal_decompose(daily_data[feature], model='additive')
        st.write("Trend Component")
        st.line_chart(result.trend)
        st.write("Seasonal Component")
        st.line_chart(result.seasonal)
        st.write("Residual Component")
        st.line_chart(result.resid)


# Correlation Matrix
elif option == 'Correlation Matrix':
    st.write("Correlation Matrix")
    corr_matrix = daily_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot()

# Statistical Anomalies
elif option == 'Statistical Anomalies':
    feature = st.selectbox('Select feature for statistical anomalies:', daily_data.columns)
    mean = daily_data[feature].mean()
    std = daily_data[feature].std()
    threshold_upper = mean + 3 * std
    threshold_lower = mean - 3 * std

    anomalies_statistical = daily_data[(daily_data[feature] > threshold_upper) | 
                                       (daily_data[feature] < threshold_lower)]
    st.write(f"Statistical Anomalies in {feature}:")
    st.dataframe(anomalies_statistical)

    
    # Plot anomalies
    st.line_chart(daily_data[feature])
    st.write("Anomalies Highlighted in Red")
    fig, ax = plt.subplots()
    ax.plot(daily_data.index, daily_data[feature], label=feature)
    ax.scatter(anomalies_statistical.index, anomalies_statistical[feature], color='red', label='Anomalies', marker='x')
    ax.legend()
    st.pyplot(fig)

    # Count the number of anomalies detected
    num_anomalies = len(anomalies_statistical)

    st.write("Number of Statistical Anomalies Detected:", num_anomalies)

# Isolation Forest Anomalies
elif option == 'Isolation Forest Anomalies':
    feature = st.selectbox('Select feature for Isolation Forest anomalies:', daily_data.columns)
    threshold = st.slider('Select threshold for Isolation Forest anomaly detection:', 0.01, 0.1, 0.01, 0.01)

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(daily_data)

    # Train the Isolation Forest model
    isolation_forest = IsolationForest(n_estimators=50, max_samples='auto', max_features=1.0, contamination=threshold, random_state=42)
    selected_feature_indices = [daily_data.columns.get_loc(col) for col in [feature]]
    pred_data = normalized_data[:, selected_feature_indices]
    anomaly_labels = isolation_forest.fit_predict(pred_data)

    # Add anomaly labels to the data
    daily_data['Anomaly_IsolationForest'] = anomaly_labels

    # Plot Isolation Forest Anomalies
    st.write(f"Anomalies Detected by Isolation Forest in {feature}:")
    st.dataframe(daily_data[daily_data['Anomaly_IsolationForest'] == -1])
    
    st.line_chart(daily_data[feature])
    st.write("Anomalies Highlighted in Red")
    fig, ax = plt.subplots()
    ax.plot(daily_data.index, daily_data[feature], label=feature)
    ax.scatter(daily_data[daily_data['Anomaly_IsolationForest'] == -1].index, 
               daily_data[daily_data['Anomaly_IsolationForest'] == -1][feature], 
               color='red', label='Anomalies', marker='x')
    ax.legend()
    st.pyplot(fig)

    # Recalculate and print the number of anomalies and normal instances
    anomaly_counts = daily_data['Anomaly_IsolationForest'].value_counts()
    num_normals_isolation_forest = anomaly_counts.get(1, 0)
    num_anomalies_isolation_forest = anomaly_counts.get(-1, 0)

    st.write(f"Number of normal instances detected by Isolation Forest: {num_normals_isolation_forest}")
    st.write(f"Number of anomalies detected by Isolation Forest: {num_anomalies_isolation_forest}")

    # Create true labels: 1 for normal, -1 for anomaly
    true_labels = np.ones(daily_data.shape[0])
    true_labels[daily_data['Anomaly_IsolationForest'] == -1] = -1

    # Get the predicted labels from Isolation Forest
    predicted_labels = daily_data['Anomaly_IsolationForest']

    
    # ROC AUC Score
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # Print ROC AUC Score
    st.write(f"ROC AUC Score: {roc_auc:.2f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels, pos_label=-1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    st.pyplot(fig)


# Autoencoder Anomalies
elif option == 'Autoencoder Anomalies':
    feature = st.selectbox('Select feature for Autoencoder anomalies:', daily_data.columns)
    percentile = st.slider('Select percentile for Autoencoder anomaly detection threshold:', 0, 100, 95, 1)

    # Normalize the data for Autoencoder
    scaler_autoencoder = MinMaxScaler()
    normalized_data_autoencoder = scaler_autoencoder.fit_transform(daily_data)

    # Define the Autoencoder model with increased layers
    input_dim_autoencoder = normalized_data_autoencoder.shape[1]
    encoding_dim_autoencoder = 14  # You can choose a different number

    autoencoder = Sequential([
        Dense(encoding_dim_autoencoder, input_shape=(input_dim_autoencoder,), activation='relu'),
        BatchNormalization(),
        Dense(encoding_dim_autoencoder // 2, activation='relu'),
        BatchNormalization(),
        Dense(encoding_dim_autoencoder // 4, activation='relu'),
        BatchNormalization(),
        Dense(encoding_dim_autoencoder // 2, activation='relu'),
        BatchNormalization(),
        Dense(input_dim_autoencoder, activation='relu')
    ])

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Setting up early stopping
    early_stopping_autoencoder = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history_autoencoder = autoencoder.fit(normalized_data_autoencoder, normalized_data_autoencoder, 
                                        epochs=100, batch_size=32, validation_split=0.2, verbose=1, 
                                        callbacks=[early_stopping_autoencoder])

    # Predict the reconstructed data
    reconstructed_data_autoencoder = autoencoder.predict(normalized_data_autoencoder)

    # Calculate the reconstruction error
    reconstruction_error_autoencoder = np.mean(np.power(normalized_data_autoencoder - reconstructed_data_autoencoder, 2), axis=1)

    # Define a threshold for anomalies (e.g., 95th percentile)
    threshold_autoencoder = np.percentile(reconstruction_error_autoencoder, 95)
    anomalies_autoencoder = daily_data[reconstruction_error_autoencoder > threshold_autoencoder]

    # Add anomaly labels to the data
    daily_data['Anomaly_Autoencoder'] = reconstruction_error_autoencoder > threshold_autoencoder


    # Print out the anomalies detected by Autoencoder
    autoencoder_anomalies = daily_data[daily_data['Anomaly_Autoencoder']]
    st.write(f"Anomalies Detected by Autoencoder in {feature}:")
    st.dataframe(daily_data[daily_data['Anomaly_Autoencoder']])

    
    
    # Plot Autoencoder Anomalies
    st.line_chart(daily_data[feature])
    st.write("Anomalies Highlighted in Red")
    fig, ax = plt.subplots()
    ax.plot(daily_data.index, daily_data[feature], label=feature)
    ax.scatter(daily_data[daily_data['Anomaly_Autoencoder']].index, 
               daily_data[daily_data['Anomaly_Autoencoder']][feature], 
               color='red', label='Anomalies', marker='x')
    ax.legend()
    st.pyplot(fig)

    # Plot Loss Function
    st.write("Autoencoder Loss Function During Training")
    plt.figure(figsize=(10, 5))
    plt.plot(history_autoencoder.history['loss'], label='Training Loss')
    plt.plot(history_autoencoder.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot()


    # Plot reconstruction error
    plt.figure(figsize=(15, 5))
    plt.plot(daily_data.index, reconstruction_error_autoencoder, label='Reconstruction Error')
    plt.axhline(y=threshold_autoencoder, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Date')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Over Time')
    plt.legend()
    st.pyplot()

    # Distribution of Reconstruction Errors
    st.write("Distribution of Reconstruction Errors")
    plt.figure(figsize=(10, 5))
    plt.hist(reconstruction_error_autoencoder, bins=50, alpha=0.75, color='blue')
    plt.axvline(threshold_autoencoder, color='red', linestyle='--', label='Threshold')
    plt.title('Distribution of Reconstruction Errors (Autoencoder)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    st.pyplot()

# LSTM Anomalies
elif option == 'LSTM Anomalies':
    feature = st.selectbox('Select feature for LSTM anomalies:', daily_data.columns)
    percentile = st.slider('Select percentile for LSTM anomaly detection threshold:', 0, 100, 95, 1)

    train_size = int(len(daily_data) * 0.7)
    test_size = len(daily_data) - train_size

    train_data = daily_data[:train_size]
    test_data = daily_data[train_size:]

    # Normalize the data
    scaler = MinMaxScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)

    # Reshape for LSTM
    X_train = np.reshape(train_data_normalized, (train_data_normalized.shape[0], train_data_normalized.shape[1], 1))
    X_test = np.reshape(test_data_normalized, (test_data_normalized.shape[0], test_data_normalized.shape[1], 1))

    # Define the LSTM model with increased layers
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(32, activation='relu'),
        Dense(train_data_normalized.shape[1])
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Setting up early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history_lstm = model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2)    


    # Predict the reconstructed data for test set
    X_test_pred = model.predict(X_test)

    # Reshape X_test_pred to match the shape of X_test
    X_test_pred = np.reshape(X_test_pred, (X_test_pred.shape[0], X_test_pred.shape[1], 1))


    # Calculate the reconstruction error for test set
    test_reconstruction_error = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

    # Define a threshold for anomalies (e.g., 95th percentile of training reconstruction error)
    threshold_test = np.percentile(test_reconstruction_error, 94.5)
    anomalies_test_autoencoder = test_reconstruction_error > threshold_test

    # Ensure that the anomalies_test_autoencoder is a 1D array
    anomalies_test_autoencoder = anomalies_test_autoencoder.ravel()

    # Add a new column to the daily_data to flag anomalies detected by LSTM
    daily_data['Anomaly_LSTM'] = False  # Initialize with False

    # Assign anomaly flags to the corresponding test data portion using .loc to avoid the warning
    daily_data.loc[daily_data.index[train_size:], 'Anomaly_LSTM'] = anomalies_test_autoencoder

    # Plot LSTM Anomalies
    st.line_chart(daily_data[feature])
    st.write("Anomalies Highlighted in Red")
    fig, ax = plt.subplots()
    ax.plot(daily_data.index, daily_data[feature], label=feature)
    ax.scatter(daily_data[daily_data['Anomaly_LSTM']].index, 
               daily_data[daily_data['Anomaly_LSTM']][feature], 
               color='red', label='Anomalies', marker='x')
    ax.legend()
    st.pyplot(fig)

    # Plot Loss Function
    st.write("LSTM Loss Function During Training")
    plt.figure(figsize=(10, 5))
    plt.plot(history_lstm.history['loss'], label='Training Loss')
    plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot()

    # Plot the reconstruction error
    st.write("LSTM Reconstruction Error")
    #plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(test_reconstruction_error, label='Reconstruction Error')
    plt.axhline(y=threshold_test, color='r', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error of LSTM Autoencoder on Test Data')
    plt.xlabel('Data Points')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.tight_layout()
    st.pyplot()

    # Distribution of Reconstruction Errors
    st.write("Distribution of Reconstruction Errors")
    plt.figure(figsize=(10, 5))
    plt.hist(test_reconstruction_error, bins=50, alpha=0.7)
    plt.axvline(threshold_test, color='red', linestyle='--', label='Threshold')
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()
