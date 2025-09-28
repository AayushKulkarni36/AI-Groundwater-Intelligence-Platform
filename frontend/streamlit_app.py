import os
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import streamlit as st
import base64
import folium
from streamlit_folium import st_folium
import numpy as np

# Optional speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Config
DEFAULT_BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:5000')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'groundwater_data.csv')

# Page config with custom theme
st.set_page_config(
    page_title='AI Groundwater Intelligence Platform',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Status cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left-color: #28a745;
    }
    
    .semi-critical-card {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left-color: #ffc107;
    }
    
    .critical-card {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left-color: #dc3545;
    }
    
    .over-exploited-card {
        background: linear-gradient(135deg, #f8d7da, #f1aeb5);
        border-left-color: #721c24;
    }
    
    .metric-large {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    /* Chat styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        display: block;
        text-align: right;
        clear: both;
        float: right;
    }
    
    .bot-message {
        background: white;
        color: #333;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        clear: both;
        float: left;
    }
    
    .message-clear {
        clear: both;
        content: "";
        display: table;
    }
    
    /* Charts container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Alert boxes */
    .alert-safe {
        background: #d1eddc;
        border: 1px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: #f8d7da;
        border: 1px solid #dc3545;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs > div > div > div > div {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1rem;
    }
    
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar for chat */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌊 AI Groundwater Intelligence Platform</h1>
    <p>Real-time monitoring • Predictive analytics • Smart decision making</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Normalize headers from demo CSV to internal keys
    rename_map = {}
    if 'Recharge (MCM)' in df.columns:
        rename_map['Recharge (MCM)'] = 'Recharge'
    if 'Extraction (MCM)' in df.columns:
        rename_map['Extraction (MCM)'] = 'Extraction'
    if 'Stage (%)' in df.columns:
        rename_map['Stage (%)'] = 'Stage'
    if rename_map:
        df = df.rename(columns=rename_map)
    df['Year'] = df['Year'].astype(int)
    return df

def get_category_color(category):
    colors = {
        'Safe': '#28a745',
        'Semi-Critical': '#ffc107',
        'Critical': '#fd7e14',
        'Over-Exploited': '#dc3545'
    }
    return colors.get(category, '#6c757d')

def create_status_card(category, stage, district, year):
    card_class = f"{category.lower().replace('-', '-').replace(' ', '-')}-card"
    color = get_category_color(category)
    
    if category == 'Safe':
        icon = "✅"
        status_text = "SAFE"
    elif category == 'Semi-Critical':
        icon = "⚠️"
        status_text = "SEMI-CRITICAL"
    elif category == 'Critical':
        icon = "🔴"
        status_text = "CRITICAL"
    else:
        icon = "🚨"
        status_text = "OVER-EXPLOITED"
    
    st.markdown(f"""
    <div class="status-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                    {icon} {status_text}
                </div>
                <div style="margin-top: 0.5rem; color: #6c757d;">
                    {district} • {year}
                </div>
            </div>
            <div style="text-align: right;">
                <div class="metric-large" style="color: {color};">{stage}%</div>
                <div class="metric-label">Stage of Development</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_alert_box(category):
    if category == 'Safe':
        st.markdown("""
        <div class="alert-safe">
            <strong>🟢 Status: SAFE</strong><br>
            Groundwater development is within sustainable limits. Continue current management practices.
        </div>
        """, unsafe_allow_html=True)
    elif category == 'Semi-Critical':
        st.markdown("""
        <div class="alert-warning">
            <strong>🟡 Status: SEMI-CRITICAL</strong><br>
            Groundwater development approaching concerning levels. Monitor closely and consider conservation measures.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-danger">
            <strong>🔴 Status: CRITICAL</strong><br>
            Immediate action required! Groundwater is being over-exploited. Implement strict conservation measures.
        </div>
        """, unsafe_allow_html=True)

def create_trend_chart(history_df):
    if history_df.empty:
        return None
    
    # Set up the plot with a professional style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Color points based on category
    colors = []
    for cat in history_df['Category']:
        colors.append(get_category_color(cat))
    
    # Plot the main line
    ax.plot(history_df['Year'], history_df['Stage'], 
            color='#667eea', linewidth=3, marker='o', markersize=8, 
            markeredgecolor='white', markeredgewidth=2, label='Stage of Development')
    
    # Color the markers by category
    scatter = ax.scatter(history_df['Year'], history_df['Stage'], 
                        c=colors, s=80, zorder=5, edgecolors='white', linewidth=2)
    
    # Add threshold lines
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Safe Threshold (70%)')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Critical Threshold (90%)')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Over-exploitation (100%)')
    
    # Add colored zones
    ax.axhspan(0, 70, alpha=0.1, color='green')
    ax.axhspan(70, 90, alpha=0.1, color='yellow')
    ax.axhspan(90, 200, alpha=0.1, color='red')
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stage of Development (%)', fontsize=12, fontweight='bold')
    ax.set_title('Groundwater Development Trend', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show integer years
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Set y-axis limits
    ax.set_ylim(0, max(110, history_df['Stage'].max() + 10))
    
    plt.tight_layout()
    return fig

def create_recharge_extraction_chart(history_df):
    if history_df.empty or 'Recharge' not in history_df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(history_df['Year']))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, history_df['Recharge'], width, 
                   label='Recharge', color='#28a745', alpha=0.8)
    bars2 = ax.bar(x + width/2, history_df['Extraction'], width, 
                   label='Extraction', color='#dc3545', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (MCM)', fontsize=12, fontweight='bold')
    ax.set_title('Recharge vs Extraction Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(history_df['Year'])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_forecast_chart(history_df, forecast_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    if not history_df.empty:
        ax.plot(history_df['Year'], history_df['Stage'], 
                color='#667eea', linewidth=3, marker='o', markersize=8,
                label='Historical Data', markeredgecolor='white', markeredgewidth=2)
    
    if not forecast_df.empty:
        ax.plot(forecast_df['Year'], forecast_df['PredictedStage'], 
                color='#ff6b6b', linewidth=3, marker='D', markersize=8,
                linestyle='--', label='Forecast', 
                markeredgecolor='white', markeredgewidth=2)
    
    # Add threshold lines
    ax.axhline(y=70, color='green', linestyle=':', alpha=0.7, label='Safe')
    ax.axhline(y=90, color='orange', linestyle=':', alpha=0.7, label='Semi-Critical')
    ax.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='Critical')
    
    # Add colored zones
    ax.axhspan(0, 70, alpha=0.1, color='green')
    ax.axhspan(70, 90, alpha=0.1, color='yellow')
    ax.axhspan(90, 200, alpha=0.1, color='red')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stage of Development (%)', fontsize=12, fontweight='bold')
    ax.set_title('Historical Data & Future Forecast', fontsize=14, fontweight='bold', pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show integer years
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    plt.tight_layout()
    return fig

def create_simulation_chart(sim_results_df):
    if sim_results_df.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Color points based on predicted category
    colors = []
    for cat in sim_results_df['PredictedCategory']:
        colors.append(get_category_color(cat))
    
    # Plot simulation results
    ax.plot(sim_results_df['Year'], sim_results_df['PredictedStage'], 
            color='#667eea', linewidth=3, marker='o', markersize=8,
            markeredgecolor='white', markeredgewidth=2)
    
    # Color the markers by predicted category
    scatter = ax.scatter(sim_results_df['Year'], sim_results_df['PredictedStage'], 
                        c=colors, s=80, zorder=5, edgecolors='white', linewidth=2)
    
    # Add colored zones
    ax.axhspan(0, 70, alpha=0.1, color='green', label='Safe Zone')
    ax.axhspan(70, 90, alpha=0.1, color='yellow', label='Semi-Critical Zone')
    ax.axhspan(90, 200, alpha=0.1, color='red', label='Critical Zone')
    
    # Add threshold lines
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stage of Development (%)', fontsize=12, fontweight='bold')
    ax.set_title('Policy Impact Simulation', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show integer years
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    plt.tight_layout()
    return fig

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Backend URL
backend_url = DEFAULT_BACKEND_URL
df = load_data()

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📈 Forecast", "🔧 What-if Simulation"])

with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 💬 Chat with AI Assistant")
        
        # Chat input
        input_col1, input_col2 = st.columns([4, 1])
        with input_col1:
            user_input = st.text_input("Ask about groundwater status...", 
                                     placeholder="e.g., How is groundwater in Pune 2022?",
                                     key="chat_input")
        
        with input_col2:
            if SR_AVAILABLE and st.button('🎙️', help="Voice input"):
                recognizer = sr.Recognizer()
                try:
                    with sr.Microphone() as source:
                        st.info('Listening... Please speak your query')
                        audio = recognizer.listen(source, phrase_time_limit=5)
                    user_input = recognizer.recognize_google(audio, language='en-IN')
                    st.success(f"Voice captured: {user_input}")
                except Exception as e:
                    st.error(f'Voice input failed: {e}')
        
        # Send button
        if st.button("Send", type="primary") and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Query backend
            with st.spinner('AI is thinking...'):
                try:
                    resp = requests.get(f"{backend_url}/query", params={'q': user_input}, timeout=20)
                    resp.raise_for_status()
                    payload = resp.json()
                    
                    answer = payload.get('answer', 'I couldn\'t find specific information for your query.')
                    st.session_state.chat_history.append({"role": "bot", "content": answer})
                    
                    # Store the latest data for visualization
                    st.session_state.latest_match = payload.get('match')
                    st.session_state.latest_history = pd.DataFrame(payload.get('history', []))
                    st.session_state.latest_predictions = pd.DataFrame(payload.get('predictions', []))
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append({"role": "bot", "content": error_msg})
            
            # Clear input and rerun to show new message
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message["role"] == "user":
                    chat_html += f'<div class="user-message">{message["content"]}</div>'
                else:
                    chat_html += f'<div class="bot-message">{message["content"]}</div>'
                chat_html += '<div class="message-clear"></div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            if 'latest_match' in st.session_state:
                del st.session_state.latest_match
            if 'latest_history' in st.session_state:
                del st.session_state.latest_history
            if 'latest_predictions' in st.session_state:
                del st.session_state.latest_predictions
            st.rerun()
    
    with col_right:
        st.markdown("### 📊 Analysis Dashboard")
        
        # Display latest results if available
        if hasattr(st.session_state, 'latest_match') and st.session_state.latest_match:
            match = st.session_state.latest_match
            create_status_card(match['Category'], match['Stage'], match['District'], match['Year'])
            create_alert_box(match['Category'])
        
        # Display charts
        if hasattr(st.session_state, 'latest_history') and not st.session_state.latest_history.empty:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Trend chart
            trend_fig = create_trend_chart(st.session_state.latest_history)
            if trend_fig:
                st.pyplot(trend_fig, use_container_width=True)
                plt.close()
            
            # Recharge vs Extraction chart
            re_fig = create_recharge_extraction_chart(st.session_state.latest_history)
            if re_fig:
                st.pyplot(re_fig, use_container_width=True)
                plt.close()
            
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📈 Future Forecast")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Forecast Parameters")
        districts_all = sorted(df['District'].unique())
        selected_district = st.selectbox('Select District', districts_all, 
                                       index=districts_all.index('Pune') if 'Pune' in districts_all else 0)
        years_ahead = st.selectbox('Forecast Period', [1,2,3,4,5], index=2)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner('Generating forecast...'):
                try:
                    resp = requests.get(f"{backend_url}/forecast", 
                                      params={'district': selected_district, 'years_ahead': years_ahead}, 
                                      timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    forecast_data = pd.DataFrame(data.get('predictions', []))
                    
                    if not forecast_data.empty:
                        st.session_state.forecast_data = forecast_data
                        st.session_state.forecast_district = selected_district
                        
                        # Show forecast summary
                        final_stage = forecast_data.iloc[-1]['PredictedStage']
                        final_category = forecast_data.iloc[-1]['PredictedCategory']
                        final_year = forecast_data.iloc[-1]['Year']
                        
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>Forecast Summary for {selected_district}</h4>
                            <div style="font-size: 2rem; color: {get_category_color(final_category)}; font-weight: bold;">
                                {final_stage}%
                            </div>
                            <div style="color: #6c757d;">
                                {final_year} • {final_category}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if final_category == 'Safe':
                            st.success("✅ Forecast indicates safe conditions")
                        elif final_category == 'Semi-Critical':
                            st.warning("⚠️ Forecast shows semi-critical levels")
                        else:
                            st.error("🚨 Forecast indicates critical conditions")
                    else:
                        st.info('No forecast data available for the selected district.')
                        
                except Exception as e:
                    st.error(f"Failed to generate forecast: {e}")
    
    with col2:
        if hasattr(st.session_state, 'forecast_data'):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Get historical data for the district
            district_history = df[df['District'].str.lower() == st.session_state.forecast_district.lower()].copy()
            district_history = district_history.sort_values('Year')
            
            # Create forecast chart
            forecast_chart = create_forecast_chart(district_history, st.session_state.forecast_data)
            if forecast_chart:
                st.pyplot(forecast_chart, use_container_width=True)
                plt.close()
            
            # Display forecast table
            st.markdown("**Detailed Forecast:**")
            st.dataframe(st.session_state.forecast_data, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 🔧 What-if Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Simulation Parameters")
        districts_all = sorted(df['District'].unique())
        sim_district = st.selectbox('Select District', districts_all, 
                                  index=districts_all.index('Pune') if 'Pune' in districts_all else 0,
                                  key="sim_district")
        
        st.markdown("**Policy Interventions:**")
        extraction_change = st.slider('Extraction Change (%)', -50, 50, 0, 
                                    help="Negative values = reduction, Positive = increase")
        recharge_change = st.slider('Recharge Enhancement (%)', -50, 50, 0,
                                  help="Positive values = artificial recharge programs")
        sim_years = st.slider('Simulation Period (years)', 1, 10, 5)
        
        # Show impact summary
        if extraction_change != 0 or recharge_change != 0:
            st.markdown("**Expected Impact:**")
            if extraction_change < 0:
                st.success(f"✅ Reduced extraction by {abs(extraction_change)}%")
            elif extraction_change > 0:
                st.warning(f"⚠️ Increased extraction by {extraction_change}%")
                
            if recharge_change > 0:
                st.success(f"✅ Enhanced recharge by {recharge_change}%")
            elif recharge_change < 0:
                st.error(f"🔴 Reduced recharge by {abs(recharge_change)}%")
        
        if st.button("Run Simulation", type="primary"):
            with st.spinner('Running simulation...'):
                try:
                    resp = requests.get(f"{backend_url}/simulate", 
                                      params={
                                          'district': sim_district,
                                          'extraction_change': extraction_change,
                                          'recharge_change': recharge_change,
                                          'years_ahead': sim_years
                                      }, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                    sim_results = pd.DataFrame(data.get('results', []))
                    
                    if not sim_results.empty:
                        st.session_state.sim_results = sim_results
                        st.session_state.sim_district_name = sim_district
                        st.session_state.sim_extraction = extraction_change
                        st.session_state.sim_recharge = recharge_change
                        
                        # Show simulation summary
                        final_stage = sim_results.iloc[-1]['PredictedStage']
                        final_category = sim_results.iloc[-1]['PredictedCategory']
                        final_year = sim_results.iloc[-1]['Year']
                        
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>Simulation Result for {sim_district}</h4>
                            <div style="font-size: 2rem; color: {get_category_color(final_category)}; font-weight: bold;">
                                {final_stage}%
                            </div>
                            <div style="color: #6c757d;">
                                {final_year} • {final_category}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if final_category == 'Safe':
                            st.success("✅ Simulation shows positive outcome")
                        elif final_category == 'Semi-Critical':
                            st.warning("⚠️ Simulation shows moderate improvement needed")
                        else:
                            st.error("🚨 Simulation indicates critical intervention required")
                        
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
    
    with col2:
        if hasattr(st.session_state, 'sim_results'):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Create simulation visualization
            sim_chart = create_simulation_chart(st.session_state.sim_results)
            if sim_chart:
                st.pyplot(sim_chart, use_container_width=True)
                plt.close()
            
            # Show intervention summary
            if hasattr(st.session_state, 'sim_extraction') and hasattr(st.session_state, 'sim_recharge'):
                district_name = getattr(st.session_state, 'sim_district_name', 'Selected District')
                st.markdown(f"""
                **Applied Interventions for {district_name}:**
                - Extraction Change: {st.session_state.sim_extraction:+d}%
                - Recharge Change: {st.session_state.sim_recharge:+d}%
                """)
            
            # Display results table
            st.markdown("**Simulation Results:**")
            st.dataframe(st.session_state.sim_results, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Quick stats sidebar (if data available)
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    if not df.empty:
        # Overall stats
        total_districts = df['District'].nunique()
        latest_year = df['Year'].max()
        
        # Category distribution for latest year
        latest_data = df[df['Year'] == latest_year]
        if not latest_data.empty:
            category_counts = latest_data['Category'].value_counts()
            
            st.markdown(f"""
            <div class="metric-box">
                <strong>Data Coverage</strong><br>
                📍 {total_districts} Districts<br>
                📅 Latest: {latest_year}
            </div>
            """, unsafe_allow_html=True)
            
            # Category breakdown
            for category, count in category_counts.items():
                color = get_category_color(category)
                percentage = (count / len(latest_data)) * 100
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                           padding: 0.5rem; margin: 0.2rem 0; border-left: 3px solid {color};">
                    <span>{category}</span>
                    <span><strong>{count}</strong> ({percentage:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
    
    # System status
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    
    # API status
    api_key_present = bool(os.environ.get('OPENAI_API_KEY'))
    model_name = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    
    if api_key_present:
        st.success(f"🤖 AI: {model_name}")
    else:
        st.warning("🤖 AI: Basic mode")
    
    # Voice input status
    if SR_AVAILABLE:
        st.success("🎙️ Voice: Available")
    else:
        st.info("🎙️ Voice: Not available")
    
    # Backend status
    try:
        resp = requests.get(f"{backend_url}/health", timeout=3)
        if resp.status_code == 200:
            st.success("🔗 Backend: Connected")
        else:
            st.error("🔗 Backend: Error")
    except:
        st.error("🔗 Backend: Offline")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p><strong>AI Groundwater Intelligence Platform</strong> | Powered by Advanced Analytics & Machine Learning</p>
    <p>🌊 Sustainable Water Management • 🔬 Data-Driven Insights • 🌍 Environmental Protection</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        Built for <strong>Smart India Hackathon 2025</strong> • Team - Runtime Rebels
    </p>
</div>
""", unsafe_allow_html=True)