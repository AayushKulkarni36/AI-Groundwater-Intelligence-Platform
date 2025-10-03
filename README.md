## Prototype: https://ai-groundwater-intelligence-platform-1.onrender.com
# AI Groundwater Intelligence Platform

A comprehensive Smart India Hackathon 2025 prototype showcasing an AI-driven chatbot and analytics platform for groundwater monitoring, forecasting, and decision support. The system integrates a Flask backend with a Streamlit frontend, designed to support sustainable water resource management through data-driven insights.

## Overview

The AI Groundwater Intelligence Platform provides government agencies, water resource managers, and policy makers with advanced tools for monitoring groundwater conditions, predicting future trends, and evaluating the impact of policy interventions. The system leverages machine learning models and natural language processing to make complex groundwater data accessible through an intuitive interface.

## Key Features

### Core Functionality
- **AI-Powered Chatbot**: Natural language interface supporting text and voice queries for groundwater status information
- **Predictive Forecasting**: Machine learning models for projecting future groundwater development stages
- **Policy Simulation**: What-if analysis tools for evaluating extraction reduction and recharge enhancement scenarios
- **Interactive Visualizations**: Real-time charts, heatmaps, and trend analysis dashboards

### Technical Capabilities
- **Multi-modal Input**: Text-based queries with optional voice recognition support
- **Real-time Analytics**: Dynamic data processing and visualization updates
- **Scalable Architecture**: Modular backend design supporting multiple data sources
- **Professional UI**: Dashboard-style interface with responsive design

## Technology Stack

### Backend Infrastructure
- **Framework**: Flask REST API
- **AI/NLP**: OpenAI GPT integration for natural language understanding
- **Machine Learning**: scikit-learn and Prophet for time series forecasting
- **Data Processing**: pandas and NumPy for data manipulation

### Frontend Interface
- **Framework**: Streamlit web application
- **Visualization**: matplotlib, plotly, and Folium for interactive charts and maps
- **UI Components**: Custom CSS styling with professional dashboard design

### Data Management
- **Format**: CSV-based demo dataset with extensible structure
- **Schema**: State, District, Year, Recharge, Extraction, Stage, Category fields
- **Compatibility**: Designed for integration with CGWB INGRES dataset

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
```   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
  
```

2. pip install -r requirements.txt
3. **Configure Environment Variables (Optional)**
   ```   # For enhanced AI capabilities
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="gpt-4o-mini"
   
   # For custom backend URL
   export BACKEND_URL="http://localhost:5000"```
4.python backend/app.py
5. In a new terminal window
   streamlit run frontend/streamlit_app.py


## Project Structure

ai-groundwater-platform/
├── backend/                    # Flask API server
│   ├── app.py                 # Main application file
│   ├── ml/                    # Machine learning modules
│   │   ├── forecasting.py     # Prediction models
│   │   └── simulation.py      # Policy simulation logic
│   └── utils/                 # Utility functions
├── frontend/                  # Streamlit web interface
│   ├── streamlit_app.py      # Main UI application
│   └── assets/               # Static assets
├── data/                     # Dataset storage
│   └── groundwater_data.csv  # Demo dataset
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore              # Git ignore rules


