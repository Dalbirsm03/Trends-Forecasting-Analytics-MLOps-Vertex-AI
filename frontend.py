import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from src.monitoring import generate_report
import os

st.set_page_config(page_title="Sales Forecasting Platform", layout="wide")

st.title("📈 Enterprise Sales Forecasting Platform")
st.markdown("Forecast sales dynamically across 171 product/region combinations using deployed ML models.")

@st.cache_data
def load_data():
    try:
        df1 = pd.read_parquet("data/processed_file/sales_featured.parquet")
        df2 = pd.read_parquet("data/processed_file/sales_feature_selected.parquet")
        # Combine to get readable labels and model features in one dataframe
        df = pd.concat([df1[['product', 'city']], df2], axis=1)
        return df, df1
    except Exception as e:
        st.error("Error loading data. Make sure you are running from the project root.")
        return pd.DataFrame(), pd.DataFrame()

df_combined, df_raw = load_data()

# Sidebar for inputs
st.sidebar.header("Forecast Settings")
model_choice = st.sidebar.selectbox("Select Model", ["xgboost", "random_forest", "linear_regression"])

st.sidebar.subheader("Select Scenario")
if not df_combined.empty:
    products = df_combined['product'].unique()
    selected_product = st.sidebar.selectbox("Product", products)
    
    cities = df_combined['city'].unique()
    selected_city = st.sidebar.selectbox("City", cities)
    
    # Filter data for dynamic defaults
    filtered_df = df_combined[(df_combined['product'] == selected_product) & (df_combined['city'] == selected_city)]
    if not filtered_df.empty:
        default_price = float(filtered_df['price_each'].mean())
        default_prod_mean = float(filtered_df['product_mean_encoded'].mean())
        default_city_mean = float(filtered_df['city_mean_encoded'].mean())
        default_sma3 = float(filtered_df['SMA_3'].mean())
        default_sma5 = float(filtered_df['SMA_5'].mean())
    else:
        default_price, default_prod_mean, default_city_mean, default_sma3, default_sma5 = 11.95, 100.5, 500.2, 150.0, 145.0
else:
    default_price, default_prod_mean, default_city_mean, default_sma3, default_sma5 = 11.95, 100.5, 500.2, 150.0, 145.0
    selected_product, selected_city = None, None

st.sidebar.subheader("Fine-Tune Features")
price_each = st.sidebar.number_input("Price Each ($)", value=default_price)
quantity = st.sidebar.number_input("Quantity Ordered", value=2.0)

with st.sidebar.expander("Advanced ML Features"):
    product_mean = st.number_input("Product Mean Encoded", value=default_prod_mean)
    city_mean = st.number_input("City Mean Encoded", value=default_city_mean)
    sma_3 = st.number_input("3-Month Moving Average", value=default_sma3)
    sma_5 = st.number_input("5-Month Moving Average", value=default_sma5)

st.sidebar.subheader("Temporal Inputs")
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])
month = st.sidebar.slider("Month", 1, 12, 1)
week = st.sidebar.slider("Week", 1, 52, 2)
year = st.sidebar.number_input("Year", value=2024)
is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Action")
    if st.button("Generate Forecast", type="primary"):
        payload = {
            "price_each": price_each,
            "product_mean_encoded": product_mean,
            "city_mean_encoded": city_mean,
            "SMA_3": sma_3,
            "SMA_5": sma_5,
            "quantity_ordered": quantity,
            "quarter": quarter,
            "month": month,
            "week": week,
            "year": int(year),
            "weekday_weekend_encoded": is_weekend
        }
        
        try:
            with st.spinner('Connecting to Model Endpoint...'):
                res = requests.post(f"http://localhost:8000/predict/{model_choice}", json=payload)
                res.raise_for_status()
                prediction = res.json()["prediction"][0]
                
            st.success(f"**Predicted Sales Volume:** ${prediction:,.2f}")
            st.session_state['last_prediction'] = prediction
            st.session_state['last_payload'] = payload
        except Exception as e:
            st.error(f"Error connecting to API. Is the FastAPI server running on port 8000? Details: {e}")

with col2:
    st.subheader("Historical Context")
    try:
        if not df_raw.empty and selected_product and selected_city:
            # Filter historical data for the selected product/city to make it highly contextual!
            hist_filtered = df_raw[(df_raw['product'] == selected_product) & (df_raw['city'] == selected_city)]
            if not hist_filtered.empty:
                monthly = hist_filtered.groupby(pd.Grouper(key='order_date', freq='ME'))['sales'].sum().reset_index()
                title = f"Historical Aggregate Sales: {selected_product} in {selected_city}"
            else:
                monthly = df_raw.groupby(pd.Grouper(key='order_date', freq='ME'))['sales'].sum().reset_index()
                title = "Historical Aggregate Sales (All Products)"
            
            fig = px.line(monthly, x='order_date', y='sales', title=title, markers=True)
            fig.update_layout(plot_bgcolor='white', xaxis_title="Date", yaxis_title="Total Sales")
            fig.update_traces(line_color='#2563eb')
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load historical data for visualization: {e}")

    if st.button("Generate Report"):
        generate_report()
        st.write("Report Generated")

