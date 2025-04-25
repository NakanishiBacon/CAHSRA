import streamlit as st
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

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
st.sidebar.header("🔍 Filters")
source_options = list(blob_map.keys()) + ["Combined"]
source = st.sidebar.selectbox("Choose data source", source_options)

if source != "Combined":
    blobs = blob_map[source]
    df_analysis = load_blob_csv(blobs["analysis"])
    df_timeseries = load_blob_csv(blobs["timeseries"])
    df_wordcloud = load_blob_csv(blobs["wordcloud"])
else:
    dfs = []
    for src, paths in blob_map.items():
        df_temp = load_blob_csv(paths["analysis"])
        df_temp["source"] = src
        dfs.append(df_temp)
    df_analysis = pd.concat(dfs, ignore_index=True)
    df_wordclouds = []
    for src, paths in blob_map.items():
        df_wc_temp = load_blob_csv(paths["wordcloud"])
        df_wordclouds.append(df_wc_temp)
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
    filtered_df = df_analysis[
        (df_analysis["date"] >= pd.to_datetime(date_range[0])) &
        (df_analysis["date"] <= pd.to_datetime(date_range[1]))
    ]
else:
    st.warning("⚠️ No usable date column found. Displaying all records.")
    filtered_df = df_analysis

# ========================
# Sentiment Bucketing
# ========================
def categorize_score(score):
    if score >= 0.8:
        return "very positive"
    elif score >= 0.5:
        return "positive"
    elif score > 0.0:
        return "light positive"
    elif score <= -0.8:
        return "very negative"
    elif score <= -0.5:
        return "negative"
    elif score < 0.0:
        return "light negative"
    else:
        return "neutral"

# ========================
# UI
# ========================
st.title(f"📊 {source} Sentiment Dashboard")

st.metric("Total Comments", len(filtered_df))


# ========================
# Correlation Heatmap
# ========================
if len(category_cols) > 1:
    st.subheader("📉 Sentiment Category Correlation")
    corr = filtered_df[category_cols].corr()
    corr.columns = [category_label_map.get(c, c) for c in corr.columns]
    corr.index = [category_label_map.get(c, c) for c in corr.index]
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto",
                         title="Category Sentiment Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# ========================
# Average Sentiment
# ========================
st.subheader("📊 Average Sentiment per Category")
avg_scores = filtered_df[category_cols].rename(columns=category_label_map).mean().reset_index()
avg_scores.columns = ['Category', 'Average Sentiment']
fig_avg = px.bar(avg_scores, x='Category', y='Average Sentiment', color='Average Sentiment',
                 labels={'Category': 'Sentiment Category'},
                 title="Mean Sentiment Score per Category", color_continuous_scale='RdYlGn')
st.plotly_chart(fig_avg, use_container_width=True)

# ========================
# Line Chart for Selected Category
# ========================
st.subheader("📈 Trend Over Time")
category_reverse_map = {v: k for k, v in category_label_map.items()}
selected_label = st.selectbox("Select category to view trend", [category_label_map[c] for c in category_cols])
selected_category = category_reverse_map[selected_label]
if 'date' in filtered_df.columns and filtered_df['date'].notna().any():
    trend = filtered_df.groupby(filtered_df['date'].dt.date)[selected_category].mean().reset_index()
    trend['date'] = pd.to_datetime(trend['date'])
    fig_trend = px.line(trend, x='date', y=selected_category,
                        labels={selected_category: selected_label, 'date': 'Date'},
                        title=f"{selected_label} - Average Sentiment Trend Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No time series available for this category.")

# ========================
# Mentions Heatmap by Day and Hour
# ========================

st.subheader("🗵️ Mentions by Day and Time")

if 'date' in filtered_df.columns and filtered_df['date'].notna().any():
    temp_df = filtered_df.copy()
    temp_df['weekday'] = temp_df['date'].dt.day_name()
    temp_df['hour'] = temp_df['date'].dt.hour

    heatmap_data = temp_df.groupby(['weekday', 'hour']).size().reset_index(name='mentions')
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['weekday'] = pd.Categorical(heatmap_data['weekday'], categories=weekdays_order, ordered=True)
    pivot_table = heatmap_data.pivot(index='weekday', columns='hour', values='mentions').fillna(0)

    fig_heatmap = px.imshow(pivot_table, 
                            labels=dict(x="Hour of Day", y="Day of Week", color="Mentions"),
                            aspect="auto",
                            title="Volume of Mentions by Day & Hour",
                            color_continuous_scale="YlGnBu")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ========================
# Word Cloud Viewer (no sentiment_score)
# ========================
st.subheader("☁️ Word Cloud Viewer")

if 'word' in df_wordcloud.columns and 'count' in df_wordcloud.columns:
    curse_words = {"shit"}
    from wordcloud import STOPWORDS
    custom_words = {"thing", "like", "people", "just", "really", "got", "youre"}
    stopwords = curse_words.union(STOPWORDS).union(custom_words)
    clean_df = df_wordcloud[~df_wordcloud['word'].str.lower().isin(stopwords)]
    word_freq = dict(zip(clean_df['word'], clean_df['count']))

    if word_freq:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        st.download_button(
            label="📅 Download Word Cloud as PNG",
            data=buf,
            file_name=f"{source.lower()}_wordcloud.png",
            mime="image/png"
        )
    else:
        st.info("No words available to generate word cloud.")
else:
    st.warning("⚠️ Word cloud file must contain 'word' and 'count' columns.")

# ========================
# Export Summary Report
# ========================
st.subheader("📄 Export Summary Report")
summary_text = f"""
Sentiment Dashboard Summary Report - {source}
Date Range: {date_range[0]} to {date_range[1] if 'date_range' in locals() else 'N/A'}
Total Comments: {len(filtered_df)}

Average Sentiment by Category:
"""
for index, row in avg_scores.iterrows():
    summary_text += f"- {row['Category']}: {row['Average Sentiment']:.3f}\n"

summary_bytes = BytesIO(summary_text.encode('utf-8'))
st.download_button(
    label="🟝 Download Text Summary",
    data=summary_bytes,
    file_name=f"{source.lower()}_sentiment_summary.txt",
    mime="text/plain"
)
