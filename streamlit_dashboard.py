import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

# ========================
# Azure Blob Setup
# ========================
AZURE_CONNECTION_STRING = st.secrets["AZURE_CONNECTION_STRING"]
CONTAINER_NAME = "visualizationdata"

@st.cache_data(ttl=86400)
def load_blob_csv(blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    blob_data = blob_client.download_blob().readall()
    return pd.read_csv(StringIO(blob_data.decode('utf-8')))

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
# Load Data
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
        df_wordclouds.append(wc_temp)
    df_analysis = pd.concat(dfs, ignore_index=True)
    df_wordcloud = pd.concat(df_wordclouds, ignore_index=True)

# ========================
# Preprocessing
# ========================
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

# ========================
# UI
# ========================
st.title(f"📊 {source} Sentiment Dashboard")
st.metric("Total Comments", len(filtered_df))
st.divider()

# ========================
# Comparison Dashboards Across Sources
# ========================
if source == "Combined":
    st.subheader("🆚 Compare Sources")
    selected_sources = st.multiselect("Select sources to compare", filtered_df['source'].unique(), default=filtered_df['source'].unique())
    comparison_df = filtered_df[filtered_df['source'].isin(selected_sources)]

# ========================
# Trend and Smoothing
# ========================
category_reverse_map = {v: k for k, v in category_label_map.items()}
selected_label = st.selectbox("Select category to view trend", [category_label_map[c] for c in category_cols])
selected_category = category_reverse_map[selected_label]

smoothing_option = st.selectbox("Smoothing", ["None", "7-Day Moving Average", "Monthly Average"])

if source == "Combined":
    trend = (comparison_df.groupby(['date', 'source'])[selected_category].mean().reset_index())
else:
    trend = filtered_df.groupby(filtered_df['date'].dt.date)[selected_category].mean().reset_index()
    trend['source'] = source

trend['date'] = pd.to_datetime(trend['date'])

if smoothing_option == "7-Day Moving Average":
    trend = trend.set_index('date').groupby('source').rolling('7D').mean().reset_index()
elif smoothing_option == "Monthly Average":
    trend = trend.set_index('date').groupby('source').resample('M').mean().reset_index()

fig_trend = px.line(trend, x='date', y=selected_category, color='source', title=f"{selected_label} - Sentiment Trend")
st.plotly_chart(fig_trend, use_container_width=True)
st.divider()

# ========================
# Advanced Stats: Skewness and Kurtosis
# ========================
st.subheader("📈 Sentiment Distribution Analysis")
selected_scores = filtered_df[selected_category].dropna()
sentiment_skew = skew(selected_scores)
sentiment_kurt = kurtosis(selected_scores)

col1, col2 = st.columns(2)
col1.metric("Skewness", f"{sentiment_skew:.3f}")
col2.metric("Kurtosis", f"{sentiment_kurt:.3f}")

fig_dist = px.histogram(selected_scores, nbins=50, marginal="violin", title=f"Sentiment Distribution for {selected_label}", labels={"value": "Sentiment Score"})
st.plotly_chart(fig_dist, use_container_width=True)
st.divider()

# ========================
# Correlation Heatmap
# ========================
if len(category_cols) > 1:
    st.subheader("📉 Sentiment Category Correlation")
    corr = filtered_df[category_cols].corr()
    corr.columns = [category_label_map.get(c, c) for c in corr.columns]
    corr.index = [category_label_map.get(c, c) for c in corr.index]
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto", title="Category Sentiment Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    csv_corr = corr.to_csv(index=True).encode('utf-8')
    st.download_button("📥 Download Correlation Matrix CSV", csv_corr, f"{source.lower()}_correlation_matrix.csv", "text/csv")
    st.divider()

# ========================
# Average Sentiment per Category
# ========================
st.subheader("📊 Average Sentiment per Category")
avg_scores = filtered_df[category_cols].rename(columns=category_label_map).mean().reset_index()
avg_scores.columns = ['Category', 'Average Sentiment']
fig_avg = px.bar(avg_scores, x='Category', y='Average Sentiment', color='Average Sentiment',
                 labels={'Category': 'Sentiment Category'}, title="Mean Sentiment Score per Category", color_continuous_scale='RdYlGn')
st.plotly_chart(fig_avg, use_container_width=True)
csv_avg = avg_scores.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Average Sentiment CSV", csv_avg, f"{source.lower()}_avg_sentiment.csv", "text/csv")
st.divider()

# ========================
# Trend Over Time
# ========================
st.subheader("📈 Trend Over Time")
category_reverse_map = {v: k for k, v in category_label_map.items()}
selected_label = st.selectbox("Select category to view trend", [category_label_map[c] for c in category_cols])
selected_category = category_reverse_map[selected_label]

trend = filtered_df.groupby(filtered_df['date'].dt.date)[selected_category].mean().reset_index()
trend['date'] = pd.to_datetime(trend['date'])
fig_trend = px.line(trend, x='date', y=selected_category, labels={selected_category: selected_label, 'date': 'Date'}, title=f"{selected_label} - Average Sentiment Trend Over Time")
st.plotly_chart(fig_trend, use_container_width=True)
csv_trend = trend.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Trend CSV", csv_trend, f"{source.lower()}_{selected_category}_trend.csv", "text/csv")
st.divider()

# ========================
# Mentions Heatmap by Day and Hour
# ========================
st.subheader("🗓️ Mentions by Day and Time")
temp_df = filtered_df.copy()
temp_df['weekday'] = temp_df['date'].dt.day_name()
temp_df['hour'] = temp_df['date'].dt.hour

heatmap_data = temp_df.groupby(['weekday', 'hour']).size().reset_index(name='mentions')
weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data['weekday'] = pd.Categorical(heatmap_data['weekday'], categories=weekdays_order, ordered=True)
pivot_table = heatmap_data.pivot(index='weekday', columns='hour', values='mentions').fillna(0)

fig_heatmap = px.imshow(pivot_table, labels=dict(x="Hour of Day", y="Day of Week", color="Mentions"),
                        aspect="auto", title="Volume of Mentions by Day & Hour", color_continuous_scale="YlGnBu")
fig_heatmap.update_layout(xaxis_title="Hour", yaxis_title="Day", xaxis_tickangle=45, font=dict(size=10))
st.plotly_chart(fig_heatmap, use_container_width=True)
csv_heatmap = pivot_table.to_csv().encode('utf-8')
st.download_button("📥 Download Mentions Heatmap CSV", csv_heatmap, f"{source.lower()}_mentions_heatmap.csv", "text/csv")
st.divider()

# ========================
# Word Cloud Viewer
# ========================
st.subheader("☁️ Word Cloud Viewer")

stopwords = set(STOPWORDS)
stopwords.update(["thing", "like", "people", "just", "really", "got", "youre", "shit"])

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
# Export Summary Report
# ========================
st.subheader("📄 Export Summary Report")
summary_text = f"""
Sentiment Dashboard Summary Report - {source}
Date Range: {date_range[0]} to {date_range[1]}
Total Comments: {len(filtered_df)}

Average Sentiment by Category:
"""
for index, row in avg_scores.iterrows():
    summary_text += f"- {row['Category']}: {row['Average Sentiment']:.3f}\n"

summary_bytes = BytesIO(summary_text.encode('utf-8'))
st.download_button(label="📥 Download Text Summary", data=summary_bytes, file_name=f"{source.lower()}_sentiment_summary.txt", mime="text/plain")
