import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import joblib # To save and load the model

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/data_for_streamlit.csv')
    return df

# Load the model (assuming it's saved as 'model.pkl')
@st.cache_resource
def load_model():
    # In a real scenario, you would save your trained model to a file
    # and load it here. For this example, we'll retrain a simple model.
    # Replace this with loading your saved model:
    # model = joblib.load('model.pkl')
    df = load_data()
    X = df[['sqft_living']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    return model

df = load_data()
model = load_model()


st.title('House Price Analysis and Prediction')

st.sidebar.header("Filter Data")

# Add input for price slider on the sidebar
min_price, max_price = float(df['price'].min()), float(df['price'].max())
price_range = st.sidebar.slider("Select Price Range", min_price, max_price, (min_price, max_price))

# Add input for bedrooms selectbox on the sidebar
all_bedrooms = sorted(df['bedrooms'].unique())
selected_bedrooms = st.sidebar.selectbox("Select Number of Bedrooms", ['All'] + all_bedrooms)


# Filter data based on price range and bedrooms
filtered_df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]
if selected_bedrooms != 'All':
    filtered_df = filtered_df[filtered_df['bedrooms'] == selected_bedrooms]


st.write("Dataset Preview:")
st.dataframe(filtered_df.head())

st.header("Exploratory Data Analysis")

# Histogram of price for filtered data
st.subheader("Distribution of House Prices (Filtered)")
fig, ax = plt.subplots()
sns.histplot(filtered_df['price'], kde=True, ax=ax)
st.pyplot(fig)

# Scatter plot of sqft_living vs price for filtered data
st.subheader("Price vs Square Footage (Filtered)")
fig, ax = plt.subplots()
sns.scatterplot(x=filtered_df['sqft_living'], y=filtered_df['price'], ax=ax)
st.pyplot(fig)

# Bar chart of bedrooms distribution for filtered data
st.subheader("Distribution of Bedrooms (Filtered)")
fig, ax = plt.subplots()
sns.countplot(x=filtered_df['bedrooms'], ax=ax)
st.pyplot(fig)


st.header("Predict House Price")

# Add input for user to predict price
sqft = st.slider("Select square footage", int(df['sqft_living'].min()), int(df['sqft_living'].max()), int(df['sqft_living'].mean()))

# Make a prediction
prediction = model.predict([[sqft]])

st.write(f"Predicted Price: ${prediction[0]:,.2f}")

# Add your visualizations and interactive elements here
