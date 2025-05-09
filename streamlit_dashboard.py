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

# ========================
st.sidebar.header("🎛️ Controls")
logo_image_map = {
    "YouTube": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/YouTube_play_button_icon_%282013%E2%80%932017%29.svg/2560px-YouTube_play_button_icon_%282013%E2%80%932017%29.svg.png",
    "Reddit": "https://www.redditstatic.com/desktop2x/img/favicon/android-icon-192x192.png",  # Updated Reddit logo URL
    "Instagram": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg",
    "Google News": "https://upload.wikimedia.org/wikipedia/commons/0/0b/Google_News_icon.png"
}

source_options = ["Combined", "YouTube", "Reddit", "Instagram", "Google News"]
labeled_options = [src for src in source_options]
label_to_source = dict(zip(labeled_options, source_options))
selected_label = st.sidebar.selectbox(
    "Choose data source",
    options=labeled_options,
    format_func=lambda x: x,
    index=source_options.index("Combined"),
    key="source_selector"
)
source = label_to_source[selected_label]

# ========================
# Load Raw Master Data
# ========================
df_youtube_master = load_blob_csv("youtube_analysis.csv")
if 'comment_published_at' in df_youtube_master.columns:
    df_youtube_master['date'] = pd.to_datetime(df_youtube_master['comment_published_at'], errors='coerce')
df_news_master = load_blob_csv("google_news_analysis.csv")
if 'timestamp' in df_news_master.columns:
    df_news_master['date'] = pd.to_datetime(df_news_master['timestamp'], errors='coerce')
df_reddit_master = load_blob_csv("reddit_analysis.csv")
if 'comment_published_at' in df_reddit_master.columns:
    df_reddit_master['date'] = pd.to_datetime(df_reddit_master['comment_published_at'], errors='coerce')
try:
    df_instagram_master = load_blob_csv("instagram_analysis.csv")
    if 'timestamp' in df_instagram_master.columns:
        df_instagram_master['date'] = pd.to_datetime(df_instagram_master['timestamp'], errors='coerce')
    else:
        st.warning("⚠️ 'scrape_timestamp' column not found in Instagram data.")
        df_instagram_master['date'] = pd.NaT
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
    if source in ["Instagram", "Google News"]:
        if source == "Instagram" and 'timestamp' in df_analysis.columns:
            df_analysis['date'] = pd.to_datetime(df_analysis['timestamp'], errors='coerce')
        elif source == "Google News" and 'timestamp' in df_analysis.columns:
            df_analysis['date'] = pd.to_datetime(df_analysis['timestamp'], errors='coerce')
        df_analysis["source"] = source
else:
    dfs = []
    for src in blob_map.keys():
        try:
            if src == "Instagram":
                temp_df = load_blob_csv("instagram_analysis.csv")
                if 'timestamp' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
            elif src == "Google News":
                temp_df = load_blob_csv("google_news_analysis.csv")
                if 'timestamp' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
            elif src == "YouTube":
                temp_df = df_youtube_master.copy()
                if 'comment_published_at' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['comment_published_at'], errors='coerce')
            elif src == "Reddit":
                temp_df = df_reddit_master.copy()
                if 'comment_published_at' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['comment_published_at'], errors='coerce')
            else:
                temp_df = load_blob_csv(blob_map[src]["analysis"])
            temp_df["source"] = src
            if src == "Instagram" and 'scrape_timestamp' in temp_df.columns:
                temp_df['date'] = pd.to_datetime(temp_df['scrape_timestamp'], errors='coerce')
            elif src == "Google News" and 'timestamp' in temp_df.columns:
                temp_df['date'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
            elif 'date' not in temp_df.columns:
                if 'comment_published_at' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['comment_published_at'], errors='coerce')
                elif 'published_at' in temp_df.columns:
                    temp_df['date'] = pd.to_datetime(temp_df['published_at'], errors='coerce')
            dfs.append(temp_df)
        except Exception as e:
            st.warning(f"⚠️ Could not load {src} data. Reason: {e}")
    df_analysis = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    

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

# ======================== # 
# Preprocessing 
# ======================== #
# Placeholder for post count display (will be moved after filtered_df is defined)
total_post_placeholder = st.empty()

# Only assign date if not already handled in Combined mode
if source != "Combined":
    if 'comment_published_at' in df_analysis.columns:
        df_analysis['date'] = pd.to_datetime(df_analysis['comment_published_at'], errors='coerce')
    elif 'published_at' in df_analysis.columns:
        df_analysis['date'] = pd.to_datetime(df_analysis['published_at'], errors='coerce')
    elif 'timestamp' in df_analysis.columns:
        df_analysis['date'] = pd.to_datetime(df_analysis['timestamp'], errors='coerce')
    else:
        df_analysis['date'] = pd.NaT
        st.sidebar.markdown(f"{platform_icons.get(plat, '')}**{plat}**: No valid dates", unsafe_allow_html=True)
    
    st.sidebar.markdown("_Note: Date range automatically spans from the oldest to most recent date available._")
    date_range = st.sidebar.date_input("Date range", [df_analysis['date'].min(), df_analysis['date'].max()])
    filtered_df = df_analysis[(df_analysis['date'] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))]
else:
    st.sidebar.markdown("_Note: Date range automatically spans from the oldest to most recent date available._")
    date_range = st.sidebar.date_input("Date range", [df_analysis['date'].min(), df_analysis['date'].max()])
    filtered_df = df_analysis[(df_analysis['date'] >= pd.to_datetime(date_range[0])) & (df_analysis['date'] <= pd.to_datetime(date_range[1]))]

if source == "Combined" and 'source' in filtered_df.columns:
    filtered_df['source'] = filtered_df['source'].astype(str)
    counts_by_source = filtered_df['source'].value_counts()
    
    icon_map = {
        "Combined": "📊",
        "YouTube": "📺",
        "Reddit": "👽",
        "Instagram": "📸",
        "Google News": "📰"
    }
    
    logo_image_map = {
        "YouTube": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/YouTube_play_button_icon_%282013%E2%80%932017%29.svg/2560px-YouTube_play_button_icon_%282013%E2%80%932017%29.svg.png",
        "Reddit": "https://upload.wikimedia.org/wikipedia/commons/en/thumb/b/bd/Reddit_Logo_Icon.svg/1024px-Reddit_Logo_Icon.svg.png",
        "Instagram": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg",
        "Google News": "https://upload.wikimedia.org/wikipedia/commons/0/0b/Google_News_icon.png"
    }

post_summary = f"\n"
if source in logo_image_map:
    post_summary += f""
else:
    pass  # Removed duplicated 📊

logo = logo_image_map.get(source)
post_summary += f"<strong style='font-size:1.2rem;'><img src='{logo}' width='22' style='vertical-align:middle; margin-right:6px;'> {source} Total Posts: {len(filtered_df):,}</strong>" if logo else f"<strong style='font-size:1.2rem;'>📊 {source} Total Posts: {len(filtered_df):,}</strong>"
post_summary += "<ul style='margin: 0; padding: 0 0 0 1.2em; list-style-type: none; font-size: 1.2rem;'>"

if source == "Combined":
    
    for platform in ['YouTube', 'Reddit', 'Instagram', 'Google News']:
        count = filtered_df[filtered_df['source'] == platform].shape[0]
        logo = logo_image_map.get(platform)
        if logo:
            post_summary += f"<li style='list-style-type:none; margin: 0 0 4px 0;'><img src='{logo}' width='18' style='vertical-align:middle; margin-right:6px;'><strong>{platform}</strong>: {count:,} posts</li>"
        else:
            post_summary += f"\n* **{platform}**: {count:,} posts\n* "
    post_summary += "</ul>"

total_post_placeholder.markdown(post_summary, unsafe_allow_html=True)

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
    with st.expander("🧮 How Often Are These Topics Mentioned?", expanded=True):
        st.markdown("This horizontal bar chart highlights the number of posts associated with each sentiment category, helping you see which topics are most frequently discussed.")
        order_choice_count = st.radio("Order bars by:", ["Alphabetical", "Value"], index=0, horizontal=True, key="category_order_count_unique")
        category_counts = filtered_df[selected_category_keys].gt(0).sum().reset_index()
        category_counts.columns = ["Category", "Count"]
        category_counts["Category"] = category_counts["Category"].map(category_label_map)
        if order_choice_count == "Value":
            category_counts = category_counts.sort_values("Count", ascending=False)
        else:
            category_counts = category_counts.sort_values("Category", ascending=True).reset_index(drop=True)
        category_counts = category_counts[::-1]
        fig_count = px.bar(
            category_counts,
            y="Category",
            x="Count",
            orientation="h",
            color="Count",
            title="Number of Mentions per Sentiment Category",
            color_continuous_scale="Blues"
        )
        fig_count.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_title='Count', yaxis_title='Category')
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
    with st.expander("😊 What’s the Overall Mood?", expanded=True):
        st.markdown("This donut chart shows the emotional makeup of the conversation — dividing sentiment into positive, neutral, and negative segments.")
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
            fig_sentiment_pie.update_layout(showlegend=False, legend_title_text='', xaxis_title='Sentiment', yaxis_title='Percentage')
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
with st.expander("🧭 Which Issues Are Viewed Most Favorably?", expanded=True):
    st.markdown("This radar chart visualizes average sentiment across categories on a scale from -1 (negative) to 1 (positive), revealing which issues are viewed more favorably.")
    radar_fig = go.Figure()
    sorted_keys = sorted(selected_category_keys, key=lambda k: category_label_map[k])
    radar_fig.add_trace(go.Scatterpolar(
        r=filtered_df[sorted_keys].mean().values,
        theta=[category_label_map[k] for k in sorted_keys],
        fill='toself',
        name='Average Sentiment'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1], title='Average Sentiment')),
        showlegend=False
    )
    radar_fig.update_traces(hovertemplate='<b>%{theta}</b><br>Sentiment=%{r:.2f}')
    st.plotly_chart(radar_fig, use_container_width=True)

# ========================
# Comment Volume
# ========================
with st.expander("📅 When Do People Talk About CAHSR the Most?", expanded=True):
    st.markdown("This time series line chart shows posting volume over time to identify peaks in public interest or major events.")
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
            fig_volume.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_title='Date', yaxis_title='Number of Posts')
            fig_volume.update_traces(line_shape="linear", hovertemplate='<b>%{x}</b><br>Posts=%{y}')
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.info("Not enough data points to generate a time series chart.")
    else:
        st.info("No valid date data available to plot volume.")

# ========================
# Sentiment Trend Over Time
# ========================
with st.expander("📈 How Has Sentiment Changed Over Time?", expanded=True):
    st.markdown("This multi-line time series chart tracks how sentiment evolves over time across different categories.")
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
            sorted_legend_labels = sorted([category_label_map[k] for k in selected_category_keys])
            fig_time_series = px.line(
                time_series.rename(columns=category_label_map),
                x='date',
                y=sorted_legend_labels,
                title=f"{trend_granularity} Sentiment Trend"
            )
            fig_time_series.update_xaxes(title_text=trend_granularity + " Date")
            fig_time_series.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, legend_title_text='', xaxis_title='Date', yaxis_title='Average Sentiment')
            fig_time_series.update_traces(hovertemplate='<b>%{x}</b><br>Sentiment=%{y:.2f}')
            st.plotly_chart(fig_time_series, use_container_width=True)
        else:
            st.info("No sentiment data available to plot trend.")
    else:
        st.info("No valid date data available to plot sentiment trend.")

# ========================
# Sentiment Momentum
# ========================
with st.expander("🚀 Where Is Sentiment Gaining or Losing Momentum?", expanded=True):
    st.markdown("This line chart illustrates the week-over-week momentum of sentiment for one selected category.")
    selected_momentum_category = st.selectbox("Choose a sentiment category to analyze momentum:", [category_label_map[k] for k in selected_category_keys], key="momentum_category_selector")
    selected_momentum_key = reverse_label_map[selected_momentum_category]
    trend_momentum_granularity = st.radio("Select time granularity:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True, key="momentum_granularity")
    if selected_category_keys:
        momentum_df = filtered_df.copy()
        momentum_df['date'] = pd.to_datetime(momentum_df['date'])
        momentum_df = momentum_df.dropna(subset=['date'])
        if momentum_df['date'].notna().any():
            if trend_momentum_granularity == "Daily":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('D'))[selected_momentum_key].mean().diff().dropna().reset_index()
            elif trend_momentum_granularity == "Monthly":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('M'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            elif trend_momentum_granularity == "Yearly":
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('Y'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            else:
                momentum_series = momentum_df.groupby(momentum_df['date'].dt.to_period('W'))[selected_category_keys[0]].mean().diff().dropna().reset_index()
            momentum_series['date'] = momentum_series['date'].dt.start_time
            momentum_series.columns = ['date', 'momentum']
            fig_momentum = px.line(momentum_series, x='date', y='momentum', title=f"Sentiment Momentum for {selected_momentum_category} ({trend_momentum_granularity})")
            fig_momentum.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_title='Date', yaxis_title='Momentum')
            fig_momentum.update_traces(hovertemplate='<b>%{x}</b><br>Momentum=%{y:.4f}')
            st.plotly_chart(fig_momentum, use_container_width=True)
        else:
            st.info("Not enough data points to generate sentiment momentum.")

# ========================
# Sentiment Distribution Analysis
# ========================
with st.expander("🧮 How Focused Are Posts on This Topic?", expanded=True):
    st.markdown("This donut chart compares the share of posts that mention a selected topic versus those that don’t.")
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
    with st.expander("🔗 Which Topics Tend to Be Mentioned Together?", expanded=True):
        st.markdown("This heatmap visualizes correlation scores between sentiment categories, revealing which issues tend to be discussed with similar sentiment.")
        sorted_corr_keys = sorted(selected_category_keys, key=lambda k: category_label_map[k])
        corr = filtered_df[sorted_corr_keys].corr()
        corr.columns = [category_label_map[c] for c in sorted_corr_keys]
        corr.index = [category_label_map[c] for c in sorted_corr_keys]
        fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto", title="Category Sentiment Correlation Matrix", labels=dict(color='Correlation'))
        st.plotly_chart(fig_corr, use_container_width=True)

# ========================
# Word Cloud Viewer
# ========================
with st.expander("☁️ What Words Stand Out the Most?", expanded=True):
    st.markdown("This word cloud displays the most frequently used words in the dataset, with larger words appearing more often.")
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
    "not", "no", "none", "never", "nothing", "nowhere", "isnt", "wasnt", "arent", "werent",

    # Social media slang / reactions
    "lol", "lmao", "omg", "bruh", "bro", "dude", "man", "girl", "guy", "idk", "ikr", "wtf", "smh",
    "yall", "ffs", "fr", "btw", "imo", "imho", "rip", "ugh", "wow", "yay", "aw", "eh",

    # Instagram/YouTube/Reddit specifics
    "post", "comment", "video", "views", "likes", "watch", "follow", "fyp", "thread", "reddit", 
    "youtube", "insta", "google", "news", "channel", "subscribe", "share", "account", "dm", 
    "reply", "click", "link", "bio", "story", "feed", "algorithm",

    # Non-content fillers
    "stuff", "everything", "something", "anything", "everyone", "someone", "somebody",

    # Added from word cloud noise
    "want", "think", "way", "done", "now", "much", "good", "see", "need", "first", "right",
    "lot", "make", "sure", "thats", "time", "projects", "cahsr", "hsr", "trains", 
    "build", "built", "building", "construction", "track", "tracks", "system", "station",
    "two", "area", "better", "public", "support", "years"
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
