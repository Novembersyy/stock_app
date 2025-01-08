import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set up the app title
st.title("Tesla Stock Price Prediction App (2023 - 2030)")

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('TSLA_stock_data.csv')  # Replace with your actual file name
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Sidebar Filters
st.sidebar.header("Filter Data")
min_year = int(data['Date'].dt.year.min())
max_year = 2030  # Extend max_year to 2030
selected_year = st.sidebar.slider("Select Year", min_year, max_year, value=min_year)
filtered_data = data[data['Date'].dt.year == selected_year]

# Display filtered data
st.subheader(f"Tesla Stock Data for {selected_year}")
st.write(filtered_data)

# Visualization of filtered data
st.subheader("Stock Price Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_data['Date'], filtered_data['Close'], label="Close Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Feature Engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
features = ['Year', 'Month', 'Open', 'High', 'Low', 'Volume']
target = 'Close'

# Train/Test Split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# User Input for Prediction
st.subheader("Make a Prediction")
year = st.number_input("Enter Year (up to 2030)", min_value=min_year, max_value=2030, value=max_year)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)
open_price = st.number_input("Enter Open Price", min_value=0.0, value=500.0)
high_price = st.number_input("Enter High Price", min_value=0.0, value=600.0)
low_price = st.number_input("Enter Low Price", min_value=0.0, value=400.0)
volume = st.number_input("Enter Volume", min_value=0.0, value=1000000.0)

# Create input data for prediction
input_data = np.array([[year, month, open_price, high_price, low_price, volume]])
future_prediction = model.predict(input_data)

st.write(f"Predicted Close Price for {year}-{month:02d}: ${future_prediction[0]:.2f}")

# Allow Visualization of Prediction
st.subheader("Visualize Prediction")
if st.checkbox("Show Prediction Point on Chart"):
    fig, ax = plt.subplots()
    ax.plot(filtered_data['Date'], filtered_data['Close'], label="Close Price", color='blue')
    if year > max_year - 1:
        ax.scatter(pd.Timestamp(year, month, 1), future_prediction[0], color='red', label="Prediction", s=100)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)
