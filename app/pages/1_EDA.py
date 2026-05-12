import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from pathlib import Path
from src.customer_support_intelligence.data import load_data, build_ticket_text

st.set_page_config(page_title="EDA Dashboard", page_icon="📊")

st.title("📊 Exploratory Data Analysis Dashboard")

# Load data
@st.cache_data
def load_eda_data():
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'sample_tickets.csv'
    df = load_data(data_path)
    df = build_ticket_text(df)
    return df

df = load_eda_data()

st.markdown("### Dataset Overview")
st.write(f"Total tickets: {len(df)}")
st.dataframe(df.head())

# Distributions
st.markdown("## 📈 Distributions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ticket Type Distribution")
    fig = px.pie(df, names='ticket_type', title="Ticket Types")
    st.plotly_chart(fig)

    st.subheader("Ticket Priority Distribution")
    fig = px.bar(df['ticket_priority'].value_counts(), title="Ticket Priorities")
    st.plotly_chart(fig)

with col2:
    st.subheader("Ticket Channel Distribution")
    fig = px.pie(df, names='ticket_channel', title="Ticket Channels")
    st.plotly_chart(fig)

    st.subheader("Customer Satisfaction Rating")
    fig = px.histogram(df, x='customer_satisfaction_rating', nbins=5, title="Satisfaction Ratings")
    st.plotly_chart(fig)

# Text Analysis
st.markdown("## 📝 Text Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Text Length Distributions")
    df['text_length'] = df['ticket_text'].str.len()
    fig = px.histogram(df, x='text_length', title="Ticket Text Length Distribution")
    st.plotly_chart(fig)

with col2:
    st.subheader("Word Count Distribution")
    df['word_count'] = df['ticket_text'].str.split().str.len()
    fig = px.histogram(df, x='word_count', title="Word Count Distribution")
    st.plotly_chart(fig)

# Word Cloud
st.subheader("Word Cloud - Ticket Descriptions")
text = ' '.join(df['ticket_text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Class Imbalance
st.markdown("## ⚖️ Class Imbalance Analysis")

st.subheader("Ticket Type Class Distribution")
type_counts = df['ticket_type'].value_counts()
fig = px.bar(x=type_counts.index, y=type_counts.values, title="Ticket Type Counts")
st.plotly_chart(fig)

st.subheader("Priority Class Distribution")
priority_counts = df['ticket_priority'].value_counts()
fig = px.bar(x=priority_counts.index, y=priority_counts.values, title="Priority Counts")
st.plotly_chart(fig)

# Interactive Dashboard Elements
st.markdown("## 📊 Interactive Insights")

# Ticket volume trends (assuming we have dates)
if 'date_of_purchase' in df.columns:
    df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], errors='coerce')
    df['month'] = df['date_of_purchase'].dt.to_period('M').astype(str)
    monthly_volume = df.groupby('month').size().reset_index(name='count')
    fig = px.line(monthly_volume, x='month', y='count', title="Monthly Ticket Volume")
    st.plotly_chart(fig)

# Satisfaction heatmap by channel and priority
if 'customer_satisfaction_rating' in df.columns:
    pivot_table = df.pivot_table(values='customer_satisfaction_rating',
                                index='ticket_channel',
                                columns='ticket_priority',
                                aggfunc='mean')
    fig = px.imshow(pivot_table, title="Satisfaction Heatmap: Channel vs Priority")
    st.plotly_chart(fig)

# Channel performance
channel_perf = df.groupby('ticket_channel').agg({
    'time_to_resolution': 'mean',
    'customer_satisfaction_rating': 'mean',
    'ticket_id': 'count'
}).round(2)
channel_perf.columns = ['Avg Resolution Time', 'Avg Satisfaction', 'Ticket Count']
st.subheader("Channel Performance")
st.dataframe(channel_perf)