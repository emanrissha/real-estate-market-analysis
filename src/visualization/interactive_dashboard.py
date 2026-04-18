"""
Interactive Streamlit Dashboard
Run with: streamlit run src/visualization/interactive_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Real Estate Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/merged_real_estate.csv')
    return df

def main():
    st.title("🏠 Real Estate Market Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_country = st.sidebar.selectbox("Select Country", ['All'] + list(df['country'].unique()))
    min_price = st.sidebar.slider("Min Price", float(df['price'].min()), float(df['price'].max()), float(df['price'].min()))
    
    # Apply filters
    filtered_df = df.copy()
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    filtered_df = filtered_df[filtered_df['price'] >= min_price]
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", len(filtered_df))
    with col2:
        st.metric("Average Price", f"${filtered_df['price'].mean():,.0f}")
    with col3:
        st.metric("Avg Satisfaction", f"{filtered_df['deal_satisfaction'].mean():.2f}/5")
    with col4:
        st.metric("Total Revenue", f"${filtered_df['price'].sum():,.0f}")
    
    # Row 1: Two charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(filtered_df, x='price', nbins=30, title="Property Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Satisfaction by Building Type")
        sat_building = filtered_df.groupby('building')['deal_satisfaction'].mean().reset_index()
        fig = px.bar(sat_building, x='building', y='deal_satisfaction', title="Average Satisfaction")
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Map
    st.subheader("Geographic Distribution")
    if 'state' in filtered_df.columns:
        state_counts = filtered_df['state'].value_counts().reset_index()
        state_counts.columns = ['state', 'count']
        fig = px.bar(state_counts.head(10), x='state', y='count', title="Top 10 States by Properties")
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Age Analysis
    st.subheader("Age Analysis")
    if 'age' in filtered_df.columns:
        fig = px.scatter(filtered_df, x='age', y='price', color='deal_satisfaction', 
                         title="Age vs Price (colored by Satisfaction)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Raw Data")
    st.dataframe(filtered_df.head(100))

if __name__ == "__main__":
    main()