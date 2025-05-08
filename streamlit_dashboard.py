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
# Sentiment Scoring Function (global use)
def score_to_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

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
    st.markdown("""
        <div style='display: flex; align-items: center;'>
            <img src='https://styles.redditmedia.com/t5_3iapt/styles/communityIcon_4iqd676dihh51.png' width='60' style='margin-right: 10px;'>
            <h1 style='margin: 0;'>CAHSR Sentiment Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
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
try:
    df_instagram_master = load_blob_csv("instagram_analysis.csv", container="visualizationdata")
except Exception as e:
    st.warning(f"⚠️ Could not load Instagram data. Reason: {e}")
    df_instagram_master = pd.DataFrame()

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

st.markdown(f"### 📊 Total Posts: {len(filtered_df):,}")

# ========================
# Standardize sentiment labels
if source == "Combined":
    if 'comment_label' not in df_analysis.columns and 'sentiment_score' in df_analysis.columns:
        df_analysis['comment_label'] = df_analysis['sentiment_score'].apply(score_to_label)
    elif 'sentiment_score' in df_analysis.columns and 'comment_label' in df_analysis.columns:
        df_analysis.loc[df_analysis['comment_label'].isna(), 'comment_label'] = df_analysis['sentiment_score'].apply(score_to_label)

# ========================
# Instagram-specific fixes
if source == "Instagram":
    if 'comment_label' not in df_analysis.columns and 'comment_sentiment' in df_analysis.columns:
        df_analysis['comment_label'] = df_analysis['comment_sentiment']
    if 'scrape_timestamp' in df_analysis.columns and 'date' not in df_analysis.columns:
        df_analysis['date'] = pd.to_datetime(df_analysis['scrape_timestamp'], errors='coerce')

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
    filtered_df = df_analysis[(df_analysis['date'] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))]
else:
    filtered_df = df_analysis

# ========================
# Category Mapping
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
if not filtered_df.empty:
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
        fig_count.update_traces(hovertemplate='<b>%{y}</b><br>Mentions=%{x}')
        st.plotly_chart(fig_count, use_container_width=True)

# ========================
# Sentiment Type Comparison
# ========================
if 'comment_label' not in filtered_df.columns and 'sentiment_score' in filtered_df.columns:
        def score_to_label(score):
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        filtered_df['comment_label'] = filtered_df['sentiment_score'].apply(score_to_label)

# Ensure no NaNs or unexpected values interfere with chart generation
# Only apply sentiment filtering if comment_label exists
if 'comment_label' in filtered_df.columns:
    filtered_df['comment_label'] = filtered_df['comment_label'].astype(str).str.lower().str.strip()
    filtered_df = filtered_df[filtered_df['comment_label'].isin(['positive', 'neutral', 'negative'])]

if 'comment_label' in filtered_df.columns:
    filtered_df['comment_label'] = filtered_df['comment_label'].astype(str).str.lower().str.strip()
    with st.expander("📊 Sentiment Type Comparison", expanded=True):
        st.markdown("This donut chart shows the percentage breakdown of positive, neutral, and negative sentiment across the selected source.")
        label_counts = filtered_df['comment_label'].value_counts().to_dict()
        expected_labels = ['positive', 'neutral', 'negative']
        sentiment_counts = pd.DataFrame({
            'Sentiment': [label.capitalize() for label in expected_labels],
            'Count': [label_counts.get(label, 0) for label in expected_labels]
        })
        if sentiment_counts['Count'].sum() > 0:
            fig_sentiment_pie = px.pie(
                sentiment_counts,
                names='Sentiment',
                values='Count',
                title="Sentiment Breakdown",
                hole=0.5,
                color='Sentiment',
                color_discrete_map={
                    'Positive': 'green',
                    'Neutral': 'gray',
                    'Negative': 'red'
                }
            )
            fig_sentiment_pie.update_layout(showlegend=False)
            fig_sentiment_pie.update_traces(
                textposition='inside',
                textinfo='percent',
                hovertemplate='<b>%{label}</b><br>Percentage=%{percent:.2%}<br>Count=%{value}',
                texttemplate='%{percent:.0%}'
            )
            st.plotly_chart(fig_sentiment_pie, use_container_width=True)
        else:
            st.info("No sentiment data available to generate the sentiment type chart.")

# ========================
# Radar View of Average Sentiment per Category
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
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=False
    )
    radar_fig.update_traces(hovertemplate='<b>%{theta}</b><br>Sentiment=%{r:.2f}')
    st.plotly_chart(radar_fig, use_container_width=True)

# ========================
# Comment Volume
# ========================
with st.expander("📆 Comment Volume", expanded=True):
    st.markdown("This chart shows the number of posts over time.")
    granularity = st.radio("Select time granularity:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True, key="volume_granularity")

    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    if filtered_df['date'].notna().any():
        if granularity == "Daily":
            volume = filtered_df.groupby(filtered_df['date'].dt.to_period('D')).size().reset_index(name='count')
        elif granularity == "Monthly":
            volume = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).size().reset_index(name='count')
        elif granularity == "Yearly":
            volume = filtered_df.groupby(filtered_df['date'].dt.to_period('Y')).size().reset_index(name='count')
        else:
            volume = filtered_df.groupby(filtered_df['date'].dt.to_period('W')).size().reset_index(name='count')
        volume['date'] = volume['date'].dt.start_time
        if len(volume) > 1:
            fig_volume = px.line(volume, x='date', y='count', title=f"{granularity} Comment Volume")
            fig_volume.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            fig_volume.update_traces(line_shape="linear", hovertemplate='<b>%{x}</b><br>Posts=%{y}')
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.info("Not enough data points to generate a time series chart.")
    else:
        st.info("No valid date data available to plot volume.")

# ========================
# Sentiment Trend Over Time
# ========================
with st.expander("📈 Sentiment Trend Over Time", expanded=True):
    st.markdown("This chart shows how public sentiment changes over time by category.")
    trend_granularity = st.radio("Select time granularity:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True, key="trend_granularity")
    trend_df = filtered_df.copy()
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    trend_df = trend_df.dropna(subset=['date'])
    if trend_df['date'].notna().any():
        if trend_granularity == "Daily":
            time_series = trend_df.groupby(trend_df['date'].dt.to_period('D'))[selected_category_keys].mean().reset_index()
        elif trend_granularity == "Monthly":
            time_series = trend_df.groupby(trend_df['date'].dt.to_period('M'))[selected_category_keys].mean().reset_index()
        elif trend_granularity == "Yearly":
            time_series = trend_df.groupby(trend_df['date'].dt.to_period('Y'))[selected_category_keys].mean().reset_index()
        else:
            time_series = trend_df.groupby(trend_df['date'].dt.to_period('W'))[selected_category_keys].mean().reset_index()

        time_series['date'] = time_series['date'].dt.start_time

        if not time_series.empty:
            fig_time_series = px.line(time_series.rename(columns=category_label_map), x='date', y=[category_label_map[k] for k in selected_category_keys], title=f"{trend_granularity} Sentiment Trend")
            fig_time_series.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, legend_title_text='')
            fig_time_series.update_traces(hovertemplate='<b>%{x}</b><br>Sentiment=%{y:.2f}')
            st.plotly_chart(fig_time_series, use_container_width=True)
        else:
            st.info("No sentiment data available to plot trend.")
    else:
        st.info("No valid date data available to plot sentiment trend.")

# ========================
# Sentiment Momentum
# ========================
with st.expander("📉 Sentiment Momentum", expanded=True):
    st.markdown("This chart shows the rate of change in sentiment over time.")
    trend_momentum_granularity = st.radio("Select time granularity:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True, key="momentum_granularity")
    if selected_category_keys:
        momentum_df = filtered_df.copy()
        momentum_df['date'] = pd.to_datetime(momentum_df['date'])
        momentum_df = momentum_df.dropna(subset=['date'])
        if momentum_df['date'].notna().any():
            if trend_momentum_granularity == "Daily":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('D'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            elif trend_momentum_granularity == "Monthly":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('M'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            elif trend_momentum_granularity == "Yearly":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('Y'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            else:
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('W'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            momentum_series['date'] = momentum_series['date'].dt.start_time
            momentum_series.columns = ['date', 'momentum']
            fig_momentum = px.line(momentum_series, x='date', y='momentum', title=f"Sentiment Momentum for {category_label_map[selected_category_keys[0]]} ({trend_momentum_granularity})")
            fig_momentum.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            fig_momentum.update_traces(hovertemplate='<b>%{x}</b><br>Momentum=%{y:.4f}')
            st.plotly_chart(fig_momentum, use_container_width=True)
        else:
            st.info("Not enough data points to generate sentiment momentum.")

# ========================
# Sentiment Distribution Analysis
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
    fig_donut.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Percentage=%{percent:.2%}<br>Count=%{value}')
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
    base_stopwords = {
    # Original terms
    "thing", "like", "people", "just", "really", "needs", "next", "says", "got", "going", "even", 
    "youre", "dont", "shit", "one", "new", "los", "san", "california", "administration", "dot", 
    "project", "highspeed", "train", "rail", "high", "speed",

    # Conjunctions / common structure
    "and", "or", "but", "so", "because", "if", "when", "while", "though", "although",

    # Filler words
    "actually", "literally", "basically", "seriously", "maybe", "kinda", "sorta", "still", 
    "already", "honestly", "anyway", "okay", "ok", "yeah", "nah",

    # Modal/helping verbs
    "can", "could", "would", "should", "will", "might", "must", "has", "have", "had", 
    "was", "were", "is", "are", "be", "being", "does", "did", "do",

    # Pronouns
    "i", "you", "he", "she", "they", "we", "it", "them", "us", "me", "my", "your", "their", "our",

    # Negations
    "not", "no", "none", "never", "nothing", "nowhere", "dont", "isnt", "wasnt", "arent", "werent",

    # Social media slang / reactions
    "lol", "lmao", "omg", "bruh", "bro", "dude", "man", "girl", "guy", "idk", "ikr", "wtf", "smh",
    "nah", "yall", "ffs", "fr", "btw", "imo", "imho", "rip", "ugh", "wow", "yay", "aw", "eh", "wow",

    # Instagram/YouTube/Reddit specifics
    "post", "comment", "video", "views", "likes", "watch", "follow", "fyp", "thread", "reddit", 
    "youtube", "insta", "google", "news", "channel", "subscribe", "share", "account", "dm", 
    "reply", "click", "link", "bio", "story", "feed", "algorithm",

    # Non-content fillers
    "thing", "stuff", "everything", "something", "anything", "nothing", "everyone", "someone", "somebody"
    }

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
# Download All Graphs
# ========================
with st.expander("🖼️ Download Visualizations", expanded=False):
    st.markdown("Download each graph individually or all at once as PNG files.")
    from plotly.io import to_image

    def download_plot(fig, filename):
        img_bytes = to_image(fig, format="png")
        st.download_button(
            label=f"📥 Download {filename}",
            data=img_bytes,
            file_name=f"{filename}.png",
            mime="image/png"
        )

    try:
            download_plot(fig_count, "count_of_posts_by_category")
    except NameError:
        st.warning("📊 'Count of Posts by Category' chart not available for download.")
    try:
            download_plot(radar_fig, "radar_category_sentiment")
    except NameError:
        st.warning("📡 Radar chart not available for download.")
    try:
            download_plot(fig_volume, "weekly_comment_volume")
    except NameError:
        st.warning("📆 Weekly Comment Volume chart not available for download.")
    try:
            download_plot(fig_time_series, "sentiment_trend_over_time")
    except NameError:
        st.warning("📈 Sentiment Trend Over Time chart not available for download.")
    try:
            download_plot(fig_momentum, "sentiment_momentum")
    except NameError:
        st.warning("📉 Sentiment Momentum chart not available for download.")
    try:
            download_plot(fig_donut, f"mention_distribution_{selected_category}")
    except NameError:
        st.warning("📈 Sentiment Distribution chart not available for download.")
    if len(selected_category_keys) > 1:
        try:
                download_plot(fig_corr, "sentiment_category_correlation")
        except NameError:
            st.warning("📉 Correlation heatmap not available for download.")

# ========================
# Export Summary Report
# ========================
with st.expander("📄 Export Summary Report", expanded=False):
    st.markdown("Generate a detailed summary of all visualized data, including raw values.")
    if not filtered_df.empty:
        from io import StringIO
        output = StringIO()

        # Category Counts
        summary_counts = filtered_df[selected_category_keys].gt(0).sum().sort_values(ascending=False)
        output.write("=== Category Mentions ===\n")
        for cat, count in summary_counts.items():
            output.write(f"{category_label_map.get(cat, cat)}: {count}\n")

        # Sentiment Distribution
        label_counts = filtered_df['comment_label'].value_counts().to_dict()
        output.write("\n=== Sentiment Breakdown ===\n")
        for label in ['positive', 'neutral', 'negative']:
            output.write(f"{label.capitalize()}: {label_counts.get(label, 0)}\n")

        # Weekly Volume
        if 'date' in filtered_df.columns and filtered_df['date'].notna().any():
            volume_df = filtered_df.groupby(filtered_df['date'].dt.to_period('W')).size().reset_index(name='post_count')
            volume_df['date'] = volume_df['date'].dt.start_time
            output.write("\n=== Weekly Comment Volume ===\n")
            for _, row in volume_df.iterrows():
                output.write(f"{row['date'].strftime('%Y-%m-%d')}: {int(row['post_count'])} posts\n")

        # Radar Sentiment Averages
        output.write("\n=== Average Sentiment by Category ===\n")
        for cat in selected_category_keys:
            avg = filtered_df[cat].mean()
            output.write(f"{category_label_map[cat]}: {avg:.3f}\n")

        # Momentum (Weekly Diff of First Category)
        output.write("\n=== Sentiment Momentum (First Category Weekly Diff) ===\n")
        try:
            momentum_df = filtered_df.dropna(subset=['date'])
            momentum_df['date'] = pd.to_datetime(momentum_df['date'])
            weekly_mean = momentum_df.groupby(momentum_df['date'].dt.to_period('W'))[selected_category_keys[0]].mean()
            momentum_series = weekly_mean.diff().dropna().reset_index()
            momentum_series['date'] = momentum_series['date'].dt.start_time
            for _, row in momentum_series.iterrows():
                output.write(f"{row['date'].strftime('%Y-%m-%d')}: {row[selected_category_keys[0]]:.4f}\n")
        except:
            output.write("Momentum data could not be computed.\n")

        # Distribution Summary
        output.write("\n=== Category Mention Distribution ===\n")
        for cat in selected_category_keys:
            val_counts = filtered_df[cat].value_counts().to_dict()
            mentioned = val_counts.get(1, 0)
            not_mentioned = val_counts.get(0, 0)
            output.write(f"{category_label_map[cat]}: Mentioned={mentioned}, Not Mentioned={not_mentioned}\n")

        # Correlation Summary
        if len(selected_category_keys) > 1:
            output.write("\n=== Sentiment Category Correlation ===\n")
            corr_matrix = filtered_df[selected_category_keys].corr()
            for row_label in corr_matrix.index:
                output.write(f"{category_label_map[row_label]} correlations:\n")
                for col_label in corr_matrix.columns:
                    output.write(f"  with {category_label_map[col_label]}: {corr_matrix.loc[row_label, col_label]:.2f}\n")

        # Word Cloud Frequencies
        try:
            output.write("\n=== Word Cloud Top Words ===\n")
            if 'word' in df_wordcloud.columns and 'count' in df_wordcloud.columns:
                clean_df = df_wordcloud.groupby('word', as_index=False)['count'].sum()
                clean_df = clean_df[~clean_df['word'].str.lower().isin(stopwords)]
                top_words = clean_df.sort_values('count', ascending=False).head(50)
                for _, row in top_words.iterrows():
                    output.write(f"{row['word']}: {int(row['count'])}\n")
            else:
                output.write("Word cloud data missing required columns.\n")
        except Exception as e:
            output.write(f"Could not compute word cloud frequencies: {e}\n")

        report = output.getvalue()
        st.text_area("Summary Preview", report, height=300)
        st.download_button("📥 Download Summary Report", data=report, file_name="summary_report.txt", mime="text/plain")
    else:
        st.info("No data available to generate summary report.")
