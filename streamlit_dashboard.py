# Import required libraries for UI, data handling, visualization, and statistics
import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go

# Set layout, title, and page icon for the Streamlit app
st.set_page_config(layout="wide", page_title="Sentiment Dashboard", page_icon="📊")

# ========================
# Azure Blob Setup
# ========================
AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
CONTAINER_NAME = "visualizationdata"

@st.cache_data(ttl=86400)
def load_blob_csv(blob_name, container=CONTAINER_NAME):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    blob_data = blob_client.download_blob().readall()
    return pd.read_csv(StringIO(blob_data.decode('utf-8')))

# ========================
# Load Raw Master Data
# ========================
df_youtube_master = load_blob_csv("youtube_master_comments.csv", container="datayoutube")
df_news_master = load_blob_csv("google_news_master_articles.csv", container="datanews")
df_reddit_master = load_blob_csv("reddit_master_comments.csv", container="datareddit")

# ========================
# Load Snapshot News Data (hidden)
# ========================
def list_snapshot_blobs():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client("snapshots")
    return [blob.name for blob in container_client.list_blobs() if blob.name.startswith("google_news_articles") and blob.name.endswith(".csv")]

snapshot_blobs = list_snapshot_blobs()
df_snapshots_combined = pd.DataFrame()
for blob_name in snapshot_blobs:
    df_temp = load_blob_csv(blob_name, container="snapshots")
    df_temp["snapshot_file"] = blob_name
    df_snapshots_combined = pd.concat([df_snapshots_combined, df_temp], ignore_index=True)

# ========================
# File Mappings by Source
# ========================
blob_map = {
    "Reddit": {
        "analysis": "reddit_analysis.csv",
        "timeseries": "reddit_time_series.csv",
        "wordcloud": "reddit_post_word_cloud.csv"
    },
    "YouTube": {
        "analysis": "youtube_analysis.csv",
        "timeseries": "youtube_time_series.csv",
        "wordcloud": "youtube_word_cloud.csv"
    },
    "Google News": {
        "analysis": "google_news_analysis.csv",
        "timeseries": "google_news_time_series.csv",
        "wordcloud": "google_news_word_cloud.csv"
    }
}

# ========================
# Sidebar: Data Source Selection and Filters
# ========================
st.sidebar.header("🎛️ Controls")
source_options = list(blob_map.keys()) + ["Combined"]
source = st.sidebar.selectbox("Choose data source", source_options)

category_label_map = {
    "category_funding_cost": "Funding Cost",
    "category_construction_progress": "Construction Progress",
    "category_politics_governance": "Politics & Governance",
    "category_environmental_impact": "Environmental Impact",
    "category_economic_impact": "Economic Impact",
    "category_alternatives_competition": "Alternatives & Competition",
    "category_regional_impact": "Regional Impact",
    "category_public_opinion": "Public Opinion",
    "category_international_comparisons": "International Comparisons"
}

# Allow users to select which sentiment categories to display in visualizations
sidebar_category_labels = st.sidebar.multiselect("Select sentiment categories to visualize", list(category_label_map.values()), default=list(category_label_map.values()))
reverse_label_map = {v: k for k, v in category_label_map.items()}
selected_category_keys = [reverse_label_map[label] for label in sidebar_category_labels]

# ========================
# Weekly Comment Volume
# ========================
with st.expander("📆 Weekly Comment Volume", expanded=True):
    st.markdown("This chart shows the number of posts each week.")
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    weekly_volume = filtered_df.groupby(filtered_df['date'].dt.to_period('W')).size().reset_index(name='count')
    weekly_volume['date'] = weekly_volume['date'].dt.start_time
    fig_volume = px.line(weekly_volume, x='date', y='count', title="Weekly Comment Volume")
    fig_volume.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_volume, use_container_width=True)

# ========================
# Category Occurrence Count
# ========================
with st.expander("📊 Count of Posts Tagged by Category", expanded=True):
    st.markdown("This chart shows how many posts were tagged with each sentiment category.")
    category_counts = filtered_df[selected_category_keys].gt(0).sum().reset_index()
    category_counts.columns = ["Category", "Count"]
    category_counts["Category"] = category_counts["Category"].map(category_label_map)
    fig_count = px.bar(category_counts, y="Category", x="Count", orientation="h", color="Count", title="Number of Mentions per Sentiment Category", color_continuous_scale="Blues")
    fig_count.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_count, use_container_width=True)

# ========================
# Average Sentiment per Category
# ========================
with st.expander("📊 Average Sentiment per Category", expanded=True):
    st.markdown("This bar chart shows the mean sentiment score per category in the selected date range.")
    avg_scores = filtered_df[selected_category_keys].mean().reset_index()
    avg_scores.columns = ['Category', 'Average Sentiment']
    avg_scores['Category'] = avg_scores['Category'].map(category_label_map)
    fig_avg = px.bar(avg_scores, y='Category', x='Average Sentiment', orientation='h', color='Category', color_discrete_sequence=px.colors.sequential.Blues)
    fig_avg.update_layout(showlegend=False, xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_avg, use_container_width=True)

# ========================
# Trend and Smoothing - Sentiment Over Time
# ========================
with st.expander("📈 Sentiment Trend Over Time", expanded=True):
    st.markdown("This chart shows how public sentiment changes over time by category.")
    category_reverse_map = {v: k for k, v in category_label_map.items()}
    trend_df = filtered_df.copy()
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    trend_df = trend_df.dropna(subset=['date'])
    time_series = trend_df.groupby(trend_df['date'].dt.to_period('W'))[selected_category_keys].mean().reset_index()
    time_series['date'] = time_series['date'].dt.start_time
    fig_time_series = px.line(time_series, x='date', y=selected_category_keys, title="Weekly Sentiment Trend")
    fig_time_series.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_time_series, use_container_width=True)

# ========================
# Sentiment Momentum
# ========================
with st.expander("📉 Sentiment Momentum", expanded=True):
    st.markdown("This chart shows the rate of change in sentiment over time.")
    if selected_category_keys:
        momentum_df = filtered_df.copy()
        momentum_df['date'] = pd.to_datetime(momentum_df['date'])
        momentum_df = momentum_df.dropna(subset=['date'])
        momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('W'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
        momentum_series['date'] = momentum_series['date'].dt.start_time
        momentum_series.columns = ['date', 'momentum']
        fig_momentum = px.line(momentum_series, x='date', y='momentum', title=f"Sentiment Momentum for {category_label_map[selected_category_keys[0]]}")
        fig_momentum.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig_momentum, use_container_width=True)

# ========================
# Sentiment Distribution Analysis
# ========================
with st.expander("📈 Sentiment Distribution Analysis", expanded=True):
    st.markdown("This histogram shows the distribution of sentiment scores for selected categories.")
    selected_scores = filtered_df[selected_category_keys[0]].dropna()
    sentiment_skew = skew(selected_scores)
    sentiment_kurt = kurtosis(selected_scores)
    col1, col2 = st.columns(2)
    col1.metric("Skewness", f"{sentiment_skew:.3f}")
    col2.metric("Kurtosis", f"{sentiment_kurt:.3f}")
    fig_dist = px.histogram(selected_scores, nbins=50, marginal="violin", title=f"Sentiment Distribution for {category_label_map[selected_category_keys[0]]}")
    st.plotly_chart(fig_dist, use_container_width=True)

# ========================
# Correlation Heatmap
# ========================
if len(selected_category_keys) > 1:
    with st.expander("📉 Sentiment Category Correlation", expanded=True):
        st.markdown("This heatmap compares how similarly sentiment scores vary across categories.")
        corr = filtered_df[selected_category_keys].corr()
        corr.columns = [category_label_map[c] for c in corr.columns]
        corr.index = [category_label_map[c] for c in corr.index]
        fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto", title="Category Sentiment Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

# ========================
# Radar Chart for Category Sentiment
# ========================
with st.expander("📡 Radar View of Average Sentiment", expanded=True):
    st.markdown("This radar chart shows average sentiment per category.")
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=avg_scores["Average Sentiment"],
        theta=avg_scores["Category"],
        fill='toself',
        name='Average Sentiment'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=False)
    st.plotly_chart(radar_fig, use_container_width=True)

# ========================
# Word Cloud Viewer
# ========================
with st.expander("☁️ Word Cloud Viewer", expanded=True):
    st.markdown("This visual displays the most frequently used words in the dataset.")
    df_wordcloud = load_blob_csv(blob_map[source]["wordcloud"] if source != "Combined" else "reddit_post_word_cloud.csv")
    custom_stopwords_input = st.text_input("Enter words to exclude from the word cloud (comma-separated):")
    custom_stopwords_list = [w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()]
    base_stopwords = {"thing", "like", "people", "just", "really", "got", "youre", "shit", "one", "new", "california", "project", "train", "high"}
    stopwords = set(STOPWORDS).union(base_stopwords).union(custom_stopwords_list)
    if 'word' in df_wordcloud.columns and 'count' in df_wordcloud.columns:
        clean_df = df_wordcloud[~df_wordcloud['word'].str.lower().isin(stopwords)]
        word_freq = dict(zip(clean_df['word'], clean_df['count']))
        if word_freq:
            wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords).generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No words available to generate word cloud.")
    else:
        st.warning("⚠️ Word cloud file must contain 'word' and 'count' columns.")

# ========================
# Export Summary Report
# ========================
with st.expander("📄 Export Summary Report", expanded=True):
    st.markdown("Download a text summary of the sentiment data.")
    summary_text = f"""
Sentiment Dashboard Summary Report - {source}
Total Comments: {len(filtered_df)}

Average Sentiment by Category:
"""
    for index, row in avg_scores.iterrows():
        line = f"- {row['Category']}: {row['Average Sentiment']:.3f}\n"
        summary_text += line
    summary_bytes = BytesIO(summary_text.encode('utf-8'))
    st.download_button(label="📥 Download Text Summary", data=summary_bytes, file_name=f"{source.lower()}_sentiment_summary.txt", mime="text/plain")
