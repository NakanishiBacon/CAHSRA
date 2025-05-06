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
st.set_page_config(layout="wide", page_title="CAHSR Sentiment Dashboard", page_icon="https://styles.redditmedia.com/t5_3iapt/styles/communityIcon_4iqd676dihh51.png")

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
    "Instagram": {
        "analysis": "instagram_analysis.csv",
        "timeseries": "instagram_time_series.csv",
        "wordcloud": ["instagram_comment_word_cloud.csv", "instagram_caption_word_cloud.csv"]
    },
    "Google News": {
        "analysis": "google_news_analysis.csv",
        "timeseries": "google_news_time_series.csv",
        "wordcloud": "google_news_word_cloud.csv"
    }
}

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
# Landing Page
# ========================
with st.container():
    st.title("🚄 CAHSR Sentiment Dashboard")
    st.markdown("""
    Welcome to the California High-Speed Rail (CAHSR) Sentiment Dashboard.
    
    This interactive dashboard aggregates and visualizes public sentiment across social and news media platforms, including Reddit, YouTube, Instagram, and Google News.

    Use the sidebar to select a data source and explore insights into funding, construction progress, environmental impact, and more.
    """)

# ========================
# Sidebar: Data Source Selection and Filters
# ========================
st.sidebar.header("🎛️ Controls")
source_options = list(blob_map.keys()) + ["Combined"]
source = st.sidebar.selectbox("Choose data source", source_options, key="source_selector")

# ========================
# Load Raw Master Data
# ========================
df_youtube_master = load_blob_csv("youtube_master_comments.csv", container="datayoutube")
df_news_master = load_blob_csv("google_news_master_articles.csv", container="datanews")
df_reddit_master = load_blob_csv("reddit_master_comments.csv", container="datareddit")
df_instagram_master = load_blob_csv("instagram_analysis.csv", container="visualizationdata")

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
# Load Selected Analysis Data
# ========================
if source != "Combined":
    blobs = blob_map[source]
    df_analysis = load_blob_csv(blobs["analysis"])
else:
    dfs = []
    for src, paths in blob_map.items():
        temp_df = load_blob_csv(paths["analysis"])
        temp_df["source"] = src
        dfs.append(temp_df)
    df_analysis = pd.concat(dfs, ignore_index=True)

# ========================
# Preprocessing
# ========================
if 'comment_published_at' in df_analysis.columns:
    df_analysis['date'] = pd.to_datetime(df_analysis['comment_published_at'], errors='coerce')
elif 'published_at' in df_analysis.columns:
    df_analysis['date'] = pd.to_datetime(df_analysis['published_at'], errors='coerce')
elif 'timestamp' in df_analysis.columns:
    df_analysis['date'] = pd.to_datetime(df_analysis['timestamp'], errors='coerce')
else:
    df_analysis['date'] = pd.NaT

if 'date' in df_analysis.columns and df_analysis['date'].notna().any():
    st.sidebar.markdown("_Note: Date range automatically spans from the oldest to most recent date available._")
    date_range = st.sidebar.date_input("Date range", [df_analysis['date'].min(), df_analysis['date'].max()])
    filtered_df = df_analysis[(df_analysis['date'] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))]
else:
    filtered_df = df_analysis


# ========================
# Category Mapping (used globally)
# ========================
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
reverse_label_map = {v: k for k, v in category_label_map.items()}
selected_category_keys = list(category_label_map.keys())

# ========================
# Count of Posts Tagged by Category
# ========================
with st.expander("📊 Count of Posts Tagged by Category", expanded=True):
    st.markdown("This chart shows how many posts were tagged with each sentiment category.")
    order_choice_count = st.radio("Order bars by:", ["Alphabetical", "Value"], horizontal=True, key="category_order_count_unique")
    category_counts = filtered_df[selected_category_keys].gt(0).sum().reset_index()
    category_counts.columns = ["Category", "Count"]
    category_counts["Category"] = category_counts["Category"].map(category_label_map)
    if order_choice_count == "Value":
        category_counts = category_counts.sort_values("Count", ascending=False)
    else:
        category_counts = category_counts.sort_values("Category")
    fig_count = px.bar(
        category_counts,
        y="Category",
        x="Count",
        orientation="h",
        color="Count",
        title="Number of Mentions per Sentiment Category",
        color_continuous_scale="Blues"
    )
    fig_count.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_count, use_container_width=True)


# ========================
# Radar Chart for Category Sentiment
# ========================
with st.expander("📡 Radar View of Average Sentiment per Category", expanded=True):
    st.markdown("This radar chart shows average sentiment per category.")
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=filtered_df[selected_category_keys].mean().values,
    theta=[category_label_map[k] for k in selected_category_keys],
        fill='toself',
        name='Average Sentiment'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=False)
    st.plotly_chart(radar_fig, use_container_width=True)

# ========================
# Weekly Comment Volume
# ========================
with st.expander("📆 Weekly Comment Volume", expanded=True):
    st.markdown("This chart shows the number of posts each week.")
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    if filtered_df['date'].notna().any():
        weekly_volume = filtered_df.groupby(filtered_df['date'].dt.to_period('W')).size().reset_index(name='count')
    else:
        weekly_volume = pd.DataFrame(columns=['date', 'count'])
    weekly_volume['date'] = weekly_volume['date'].dt.start_time
    fig_volume = px.line(weekly_volume, x='date', y='count', title="Weekly Comment Volume")
    fig_volume.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    fig_volume.update_traces(line_shape="spline")
    st.plotly_chart(fig_volume, use_container_width=True)

# ========================
# Sentiment Trend Over Time
# ========================
with st.expander("📈 Sentiment Trend Over Time", expanded=True):
    st.markdown("This chart shows how public sentiment changes over time by category.")
    trend_df = filtered_df.copy()
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    trend_df = trend_df.dropna(subset=['date'])
    if trend_df['date'].notna().any():
        time_series = trend_df.groupby(trend_df['date'].dt.to_period('W'))[selected_category_keys].mean().reset_index()
    else:
        time_series = pd.DataFrame(columns=['date'] + selected_category_keys)
    time_series['date'] = time_series['date'].dt.start_time
    fig_time_series = px.line(time_series.rename(columns=category_label_map), x='date', y=[category_label_map[k] for k in selected_category_keys], title="Weekly Sentiment Trend")
    fig_time_series.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, legend_title_text='')
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
        if momentum_df['date'].notna().any():
            momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('W'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
        else:
            momentum_series = pd.DataFrame(columns=['date', 'momentum'])
        momentum_series['date'] = momentum_series['date'].dt.start_time
        momentum_series.columns = ['date', 'momentum']
        fig_momentum = px.line(momentum_series, x='date', y='momentum', title=f"Sentiment Momentum for {category_label_map[selected_category_keys[0]]}")
        fig_momentum.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig_momentum, use_container_width=True)

# ========================
# Sentiment Distribution Analysis (Donut Chart)
# ========================
with st.expander("📈 Sentiment Distribution Analysis", expanded=True):
    st.markdown("This chart shows the proportion of posts that mention vs. don't mention the selected category.")
    selected_category_label = st.selectbox("Choose a sentiment category to view:", list(category_label_map.values()), key="distribution_category_selector")
    selected_category = reverse_label_map[selected_category_label]
    counts = filtered_df[selected_category].value_counts().sort_index()
    donut_df = pd.DataFrame({
        "Mentioned": ["No", "Yes"],
        "Count": [counts.get(0, 0), counts.get(1, 0)]
    })
    fig_donut = px.pie(
        donut_df,
        names="Mentioned",
        values="Count",
        title=f"Mention Proportion for {category_label_map[selected_category]}",
        hole=0.5
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(showlegend=True, legend_title_text="")
    st.plotly_chart(fig_donut, use_container_width=True)

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
# Word Cloud Viewer
# ========================
with st.expander("☁️ Word Cloud Viewer", expanded=True):
    st.markdown("This visual displays the most frequently used words in the dataset.")
    wordcloud_files = blob_map[source]["wordcloud"] if source != "Combined" else "reddit_post_word_cloud.csv"
    df_wordcloud = pd.DataFrame()
    if isinstance(wordcloud_files, list):
        for wc_file in wordcloud_files:
            df_temp = load_blob_csv(wc_file)
            df_wordcloud = pd.concat([df_wordcloud, df_temp], ignore_index=True)
    else:
        try:
            df_wordcloud = load_blob_csv(wordcloud_files)
        except Exception as e:
            if source == "YouTube":
                df_wordcloud = pd.read_csv("/mnt/data/youtube_word_cloud.csv")
            else:
                st.warning(f"⚠️ Could not load word cloud file for {source}. Reason: {str(e)}")
                df_wordcloud = pd.DataFrame()
        except Exception as e:
            st.warning(f"⚠️ Could not load word cloud file for {source}. Reason: {str(e)}")
            df_wordcloud = pd.DataFrame()

    custom_stopwords_input = st.text_input("Enter words to exclude from the word cloud (comma-separated):")
    custom_stopwords_list = [w.strip().lower() for w in custom_stopwords_input.split(",") if w.strip()]
    base_stopwords = {"thing", "like", "people", "just", "really", "got", "youre", "shit", "one", "new", "california", "project", "train", "high"}
    stopwords = set(STOPWORDS).union(base_stopwords).union(custom_stopwords_list)

    if 'word' in df_wordcloud.columns and 'count' in df_wordcloud.columns:
        clean_df = df_wordcloud.groupby('word', as_index=False)['count'].sum()
        clean_df = clean_df[~clean_df['word'].str.lower().isin(stopwords)]
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
"""
    summary_bytes = BytesIO(summary_text.encode('utf-8'))
    st.download_button(
        label="📥 Download Text Summary",
        data=summary_bytes,
        file_name=f"{source.lower()}_sentiment_summary.txt",
        mime="text/plain"
    )
