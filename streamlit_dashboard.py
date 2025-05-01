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

st.set_page_config(layout="wide", page_title="Sentiment Dashboard", page_icon="📊")


# ========================
# Azure Blob Setup
# ========================
# Set up connection to Azure Blob Storage
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
# Load raw master data from YouTube, News, and Reddit containers
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

# Load all matching snapshot CSVs from the 'snapshots' container
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
# Sidebar: Data Source Selection
# ========================
st.sidebar.header("🎛️ Controls")
source_options = list(blob_map.keys()) + ["Combined"]
source = st.sidebar.selectbox("Choose data source", source_options)

# ========================
# Safety Check and Basic UI
# ========================
if not source:
    st.warning("Please select a data source from the sidebar.")
    st.stop()

st.title("📊 Sentiment Analysis Dashboard")
st.markdown("This dashboard visualizes sentiment trends across Reddit, YouTube, and Google News.")
st.info(f"You are viewing: **{source}** data")

# ========================
# Load Selected Analysis Data
# ========================
if source != "Combined":
    blobs = blob_map[source]
    df_analysis = load_blob_csv(blobs["analysis"])
    df_wordcloud = load_blob_csv(blobs["wordcloud"])
else:
    dfs = []
    df_wordclouds = []
    for src, paths in blob_map.items():
        temp_df = load_blob_csv(paths["analysis"])
        temp_df["source"] = src
        dfs.append(temp_df)
        wc_temp = load_blob_csv(paths["wordcloud"])
        wc_temp["source"] = src
        df_wordclouds.append(wc_temp)
    df_analysis = pd.concat(dfs, ignore_index=True)
    df_wordcloud = pd.concat(df_wordclouds, ignore_index=True)

# ========================
# Preprocessing
# ========================
# Preprocess the date column by checking which datetime field exists
if 'comment_published_at' in df_analysis.columns:
    df_analysis['date'] = pd.to_datetime(df_analysis['comment_published_at'], errors='coerce')
elif 'published_at' in df_analysis.columns:
    df_analysis['date'] = pd.to_datetime(df_analysis['published_at'], errors='coerce')
else:
    df_analysis['date'] = pd.NaT

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

category_cols = [col for col in df_analysis.columns if col in category_label_map]

if 'date' in df_analysis.columns and df_analysis['date'].notna().any():
    date_range = st.sidebar.date_input("Date range", [df_analysis['date'].min(), df_analysis['date'].max()])
    filtered_df = df_analysis[(df_analysis['date'] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))]
else:
    st.warning("⚠️ No usable date column found. Displaying all records.")
    filtered_df = df_analysis

if filtered_df.empty:
    st.warning("⚠️ No comments available for the selected date range.")
    st.stop()

# Continue with: UI metrics, trend line, smoothing, skew/kurtosis, correlation heatmap, category averages, word cloud, heatmap, summary
# ========================
# UI Metrics
# ========================
st.metric("Total Comments", len(filtered_df))
st.divider()

# ========================
# Category Occurrence Count
# ========================
st.subheader("📊 Count of Posts Tagged by Category")
# Count how many posts are tagged with each sentiment category
category_counts = filtered_df[category_cols].gt(0).sum().reset_index()
category_counts.columns = ["Category", "Count"]
category_counts["Category"] = category_counts["Category"].map(category_label_map)
fig_count = px.bar(category_counts, x="Category", y="Count", color="Count",
                   title="Number of Mentions per Sentiment Category", color_continuous_scale="Blues")
fig_count.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig_count, use_container_width=True)
st.divider()

# ========================
# Average Sentiment per Category
# ========================
st.subheader("📊 Average Sentiment per Category")
# Calculate average sentiment score for each category
avg_scores = filtered_df[category_cols].rename(columns=category_label_map).mean().reset_index()
avg_scores.columns = ['Category', 'Average Sentiment']
fig_avg = px.bar(avg_scores, x='Category', y='Average Sentiment', color='Category', color_discrete_sequence=px.colors.sequential.Blues)
fig_avg.update_layout(
    showlegend=False,
    title="Mean Sentiment Score per Category",
    xaxis_title="Sentiment Category",
    yaxis_title="Average Sentiment"
)

st.plotly_chart(fig_avg, use_container_width=True)
st.divider()

# ========================
# Trend and Smoothing
# ========================
st.subheader("📈 Sentiment Trend Over Time")
category_reverse_map = {v: k for k, v in category_label_map.items()}

multi_select_mode = st.toggle("Compare multiple categories", value=False)

if multi_select_mode:
    selected_labels = st.multiselect("Select categories to compare", [category_label_map[c] for c in category_cols], default=[category_label_map[category_cols[0]]])
    selected_categories = [category_reverse_map[label] for label in selected_labels]
else:
    selected_label = st.selectbox("Select category to view trend", [category_label_map[c] for c in category_cols], key="trend_category_select")
    selected_categories = [category_reverse_map[selected_label]]

smoothing_option = st.selectbox("Smoothing", ["None", "7-Day Moving Average", "Monthly Average"])

# Prepare trend data with optional smoothing and category comparison
trend_df = filtered_df.copy()
trend_df['date'] = pd.to_datetime(trend_df['date'])
trend_df = trend_df.dropna(subset=['date'])

trend_lines = []
for cat in selected_categories:
    temp = trend_df.copy()
    if source != "Combined":
        temp['source'] = source
    grouped = temp.groupby([temp['date'].dt.date, 'source'])[cat].mean().reset_index(name='value')
    grouped['date'] = pd.to_datetime(grouped['date'])
    grouped['category'] = category_label_map[cat]
    trend_lines.append(grouped)

trend = pd.concat(trend_lines, ignore_index=True)

if smoothing_option == "7-Day Moving Average":
    trend = trend.set_index('date').groupby(['category', 'source']).rolling('7D').mean().reset_index()
elif smoothing_option == "Monthly Average":
    trend = trend.set_index('date').groupby(['category', 'source']).resample('M').mean().reset_index()

fig_trend = px.line(trend, x='date', y='value', color='category' if multi_select_mode else 'source',
                    title="Sentiment Trend Over Time")
st.plotly_chart(fig_trend, use_container_width=True)
st.divider()

# ========================
# Advanced Stats: Skewness and Kurtosis
# ========================
st.subheader("📈 Sentiment Distribution Analysis")
if multi_select_mode:
    st.info("📌 Skewness & kurtosis shown for first selected category.")
    selected_scores = filtered_df[selected_categories[0]].dropna()
    display_label = category_label_map[selected_categories[0]]
else:
    selected_scores = filtered_df[selected_categories[0]].dropna()
    display_label = selected_label

# Calculate advanced statistics: skewness and kurtosis
sentiment_skew = skew(selected_scores)
sentiment_kurt = kurtosis(selected_scores)

col1, col2 = st.columns(2)
col1.metric("Skewness", f"{sentiment_skew:.3f}")
col2.metric("Kurtosis", f"{sentiment_kurt:.3f}")

fig_dist = px.histogram(selected_scores, nbins=50, marginal="violin", title=f"Sentiment Distribution for {display_label}", labels={"value": "Sentiment Score"})
st.plotly_chart(fig_dist, use_container_width=True)
st.divider()

# ========================
# Correlation Heatmap
# ========================
if len(category_cols) > 1:
    st.subheader("📉 Sentiment Category Correlation")
    # Compute correlation matrix between sentiment categories
    corr = filtered_df[category_cols].corr()
    corr.columns = [category_label_map.get(c, c) for c in corr.columns]
    corr.index = [category_label_map.get(c, c) for c in corr.index]
    fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto", title="Category Sentiment Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
st.divider()

# ========================
# Word Cloud Viewer
# ========================
st.subheader("☁️ Word Cloud Viewer")

# User-defined stopwords input
# Allow users to input their own stopwords to exclude from the word cloud
custom_stopwords_input = st.text_input("Enter words to exclude from the word cloud (comma-separated):")
custom_stopwords_list = [w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()]

# Hardcoded base stopwords
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

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.download_button(label="📥 Download Word Cloud PNG", data=buf, file_name=f"{source.lower()}_wordcloud.png", mime="image/png")
    else:
        st.info("No words available to generate word cloud.")
else:
    st.warning("⚠️ Word cloud file must contain 'word' and 'count' columns.")
st.divider()

# ========================
# Time Series by Category (Individual Over Time)
# ========================
st.subheader("📈 Time Series for Each Category")
time_range = st.date_input("Select date range for time series", [filtered_df['date'].min(), filtered_df['date'].max()], key="time_series_range")
time_series_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(time_range[0])) & (filtered_df['date'] <= pd.to_datetime(time_range[1]))].copy()
time_series_df['date'] = pd.to_datetime(time_series_df['date'])
time_series_df = time_series_df.dropna(subset=['date'])
time_series = time_series_df.groupby(time_series_df['date'].dt.to_period('W'))[category_cols].mean().reset_index()
time_series['date'] = time_series['date'].dt.start_time
fig_time_series = px.line(time_series, x='date', y=category_cols, title="Weekly Average Sentiment per Category")
st.plotly_chart(fig_time_series, use_container_width=True)

# ========================
# Additional Visualizations
# ========================

# Boxplot of Sentiment Score by Source
st.subheader("📦 Sentiment Score Distribution by Source")
if source == "Combined" and len(selected_categories) == 1:
    fig_box = px.box(filtered_df, x='source', y=selected_categories[0], points='all', title="Sentiment Score Distribution by Source")
    st.plotly_chart(fig_box, use_container_width=True)

# Weekly Comment Volume with Date Filter
st.subheader("📆 Weekly Comment Volume")
volume_range = st.date_input("Select date range for volume chart", [filtered_df['date'].min(), filtered_df['date'].max()], key="volume_date_range")
filtered_volume_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(volume_range[0])) & (filtered_df['date'] <= pd.to_datetime(volume_range[1]))]
weekly_volume = filtered_volume_df.groupby(filtered_volume_df['date'].dt.to_period('W')).size().reset_index(name='count')
weekly_volume['date'] = weekly_volume['date'].dt.start_time
fig_volume = px.line(weekly_volume, x='date', y='count', title="Weekly Comment Volume")
st.plotly_chart(fig_volume, use_container_width=True)

# ========================
# Export Summary Report
# ========================

# Export a summary report containing the number of comments and average sentiment per category
st.subheader("📄 Export Summary Report")
summary_text = f"""
Sentiment Dashboard Summary Report - {source}
Date Range: {date_range[0]} to {date_range[1]}
Total Comments: {len(filtered_df)}

Average Sentiment by Category:
"""
for index, row in avg_scores.iterrows():
    line = f"- {row['Category']}: {row['Average Sentiment']:.3f}"
    summary_text += line

summary_bytes = BytesIO(summary_text.encode('utf-8'))
st.download_button(label="📥 Download Text Summary", data=summary_bytes, file_name=f"{source.lower()}_sentiment_summary.txt", mime="text/plain")
