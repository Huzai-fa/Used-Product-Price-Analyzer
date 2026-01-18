# app.py - Complete Working Version
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Used Product Price Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .price-highlight {
        font-size: 2.5rem;
        font-weight: bold;
        color: #10B981;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stSelectbox, .stTextInput, .stNumberInput {
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #3B82F6;
    }
    .warning-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">💰 Used Product Price Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get AI-powered price predictions for your used items based on market data</p>', unsafe_allow_html=True)

# Initialize session state for form persistence
if 'product_data' not in st.session_state:
    st.session_state.product_data = {
        'title': "",
        'condition': 'Good',
        'brand': 'Apple',
        'product_year': 2021,
        'shipping_cost': 0.0,
        'has_defects': False
    }

# Sidebar for input
with st.sidebar:
    st.markdown("### 🛒 Used Product Price Prediction")
    st.markdown("Data Powered by Ebay")
    
    # Product Title
    product_title = ""
    
    # Condition
    condition = ""
    
    # Brand
    brand = ""
    
    # Year and Shipping
    col1, col2 = st.columns(2)
    with col1:
        product_year = st.number_input(
            "**Year**",
            min_value=1990,
            max_value=datetime.now().year,
            value=st.session_state.product_data['product_year'],
            help="Year of manufacture or release"
        )
    
    with col2:
        shipping_cost = st.number_input(
            "**Shipping Cost ($)**",
            min_value=0.0,
            value=st.session_state.product_data['shipping_cost'],
            step=5.0,
            help="Estimated shipping cost to include"
        )
    
    # Defects
    has_defects = st.checkbox(
        "**Has visible defects**",
        value=st.session_state.product_data['has_defects'],
        help="Check if there are cracks, scratches, or non-working parts"
    )
    
    # Additional details
  
    
    st.divider()
    
    # Predict button
    predict_button = st.button(
        "🚀 GET THE RESULTS FROM MODEL",
        type="primary",
        use_container_width=True,
        help="Click to analyze and predict the market price"
    )
    
    # Save current form state
    st.session_state.product_data = {
        'title': product_title,
        'condition': condition,
        'brand': brand,
        'product_year': product_year,
        'shipping_cost': shipping_cost,
        'has_defects': has_defects
    }
    
    st.divider()
    
    # Quick examples
   
    
    # Model status check
    st.divider()
    st.markdown("### ⚙️ System Status")
    
    # Check API status
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=2)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get('model_loaded'):
                st.success("✅ Model loaded and ready")
            else:
                st.warning("⚠️ Using fallback mode")
        else:
            st.error("❌ API not responding")
    except:
        st.error("❌ API server not running")
        st.info("Start API: `uvicorn api:app --reload`")

# Main content area
if predict_button:
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress
    for i in range(1, 101, 20):
        progress_bar.progress(i)
        status_text.text(f"Analyzing market data... {i}%")
    
    # Prepare API request
    product_data = {
        "title": product_title,
        "condition": condition.lower(),
        "brand": brand if brand else None,
        "product_year": product_year,
        "shipping_cost": shipping_cost,
        "has_defects": has_defects
    }
    
    try:
        # Make API call
        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=product_data, timeout=15)
        
        # Complete progress bar
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        if response.status_code == 200:
            result = response.json()
            
            # Display main results
            st.markdown("## 📊 Price Prediction Results")
            
            # Main metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #6B7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">PREDICTED PRICE</h3>
                    <div class="price-highlight">${:,.2f}</div>
                    <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                        Confidence: {:.0f}%
                    </p>
                </div>
                """.format(result['predicted_price'], result['confidence']*100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #6B7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">PRICE RANGE</h3>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #3B82F6;">
                        ${:,.2f} - ${:,.2f}
                    </div>
                    <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                        Expected market range
                    </p>
                </div>
                """.format(result['price_range_low'], result['price_range_high']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #6B7280; margin: 0 0 0.5rem 0; font-size: 0.9rem;">SELLING TIP</h3>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #10B981;">
                        List at: ${:,.2f}
                    </div>
                    <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                        Best listing price
                    </p>
                </div>
                """.format(result['predicted_price']), unsafe_allow_html=True)
            
            # Show warning if using fallback
            if "fallback" in result.get('message', '').lower():
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Note:</strong> Using fallback predictions. For more accurate results, 
                    train the model by running: <code>python main.py</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Success:</strong> AI model prediction based on market data analysis
                </div>
                """, unsafe_allow_html=True)
            
            # Price visualization
            st.markdown("## 📈 Price Analysis")
            
            # Create price distribution chart
            fig = go.Figure()
            
            # Add range bar
            fig.add_trace(go.Bar(
                name='Price Range',
                x=['Estimated Value'],
                y=[result['price_range_high'] - result['price_range_low']],
                base=result['price_range_low'],
                marker_color='rgba(59, 130, 246, 0.7)',
                width=0.4,
                hovertemplate='Range: $%{base:.2f} - $%{y:.2f}<extra></extra>'
            ))
            
            # Add predicted price line
            fig.add_trace(go.Scatter(
                name='Predicted Price',
                x=['Estimated Value'],
                y=[result['predicted_price']],
                mode='markers+text',
                marker=dict(size=20, color='#EF4444', symbol='diamond'),
                text=[f"${result['predicted_price']:.2f}"],
                textposition="top center",
                hovertemplate='Predicted: $%{y:.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title="Price Estimate Visualization",
                yaxis_title="Price ($)",
                showlegend=True,
                height=450,
                plot_bgcolor='rgba(240, 242, 246, 0.8)',
                paper_bgcolor='white',
                font=dict(family="Arial", size=12, color="#1F2937")
            )
            
            # Add range annotations
            fig.add_annotation(
                x=0,
                y=result['price_range_low'],
                text=f"Low: ${result['price_range_low']:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
            
            fig.add_annotation(
                x=0,
                y=result['price_range_high'],
                text=f"High: ${result['price_range_high']:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=40
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action recommendations
            st.markdown("## 💡 Selling Recommendations")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.markdown("""
                ### 🎯 **List Now**
                **Price:** ${:,.2f}
                
                - Quick sale price
                - High chance of sale
                - Good for fast turnover
                """.format(result['predicted_price'] * 0.95))
            
            with rec_col2:
                st.markdown("""
                ### 📊 **Wait for Offers**
                **Price:** ${:,.2f}
                
                - Best value price
                - Room for negotiation
                - Target serious buyers
                """.format(result['predicted_price']))
            
            with rec_col3:
                st.markdown("""
                ### ⏳ **Patience Pays**
                **Price:** ${:,.2f}
                
                - Maximum value
                - Wait for right buyer
                - Best for rare items
                """.format(result['predicted_price'] * 1.1))
            
            # Marketplace comparison
            st.markdown("## 🏪 Marketplace Comparison")
            
            # Simulated marketplace prices
            marketplaces = {
                "eBay": result['predicted_price'] * 0.95,
              
            }
            
            # Create marketplace comparison chart
            market_df = pd.DataFrame({
                'Marketplace': list(marketplaces.keys()),
                'Estimated Price': list(marketplaces.values()),
                'Difference': [p - result['predicted_price'] for p in marketplaces.values()]
            })
            
            # Display as bar chart
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=market_df['Marketplace'],
                y=market_df['Estimated Price'],
                marker_color=['#FF9900', '#1877F2', '#6C3', '#00A2FF'],
                text=[f"${p:,.0f}" for p in market_df['Estimated Price']],
                textposition='auto',
            ))
            
            # Add our prediction line
            fig2.add_hline(
                y=result['predicted_price'],
                line_dash="dash",
                line_color="red",
                annotation_text="Our Prediction",
                annotation_position="top right"
            )
            
            fig2.update_layout(
                title="Estimated Prices on Different Platforms",
                yaxis_title="Price ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show raw features (optional)
          
            
            # Copy results button
            st.divider()
            copy_col1, copy_col2, copy_col3 = st.columns([1, 2, 1])
            with copy_col2:
                if st.button("📋 Copy Results to Clipboard", use_container_width=True):
                    results_text = f"""
                    Price Prediction for {product_title}:
                    - Predicted Price: ${result['predicted_price']:.2f}
                    - Price Range: ${result['price_range_low']:.2f} - ${result['price_range_high']:.2f}
                    - Confidence: {result['confidence']*100:.0f}%
                    - Condition: {condition}
                    """
                    st.code(results_text, language="text")
                    st.success("Results copied! (Select and copy the text above)")
        
        elif response.status_code == 503:
            st.error("## ⚠️ Model Not Trained")
            st.markdown("""
            The AI model needs to be trained before making predictions.
            
            **To train the model:**
            
            1. **Open a new terminal** (don't close this one)
            2. **Navigate to your project folder:**
               ```bash
               cd used-price-analyzer
               ```
            3. **Activate virtual environment:**
               - **Windows:** `venv\\Scripts\\activate`
               - **Mac/Linux:** `source venv/bin/activate`
            4. **Train the model:**
               ```bash
               python main.py
               ```
            5. **Wait for training to complete** (1-2 minutes)
            6. **Refresh this page** and try again
            
            *In the meantime, you can use our fallback pricing below:*
            """)
            
            # Fallback pricing estimate
            st.markdown("### 💰 Fallback Price Estimate")
            
            # Simple rule-based fallback
            base_price = 500 if "iphone" in product_title.lower() else 300
            if "pro" in product_title.lower():
                base_price *= 1.3
            if "max" in product_title.lower():
                base_price *= 1.5
            
            # Condition multiplier
            condition_mult = {
                "new": 1.2, "like new": 1.1, "excellent": 1.05,
                "very good": 1.0, "good": 0.9, "acceptable": 0.8,
                "used": 0.7, "fair": 0.6, "poor": 0.5
            }
            
            cond_lower = condition.lower()
            multiplier = 0.8
            for key, value in condition_mult.items():
                if key in cond_lower:
                    multiplier = value
                    break
            
            fallback_price = base_price * multiplier
            fallback_low = fallback_price * 0.8
            fallback_high = fallback_price * 1.2
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Price", f"${fallback_price:.2f}")
            with col2:
                st.metric("Price Range", f"${fallback_low:.2f} - ${fallback_high:.2f}")
            with col3:
                st.metric("Confidence", "50%")
            
        else:
            st.error(f"## API Error {response.status_code}")
            st.code(f"Response: {response.text}")
            st.info("""
            **Troubleshooting steps:**
            1. Check if API server is running: `uvicorn api:app --reload`
            2. Make sure port 8000 is not blocked
            3. Check API logs for detailed error messages
            """)
    
    except requests.exceptions.ConnectionError:
        st.error("## ❌ Cannot Connect to API")
        st.markdown("""
        The price prediction API is not responding. Here's how to fix it:
        
        ### **Step 1: Start the API Server**
        
        **Open a NEW terminal window** and run:
        
        ```bash
        # Navigate to project folder
        cd used-price-analyzer
        
        # Activate virtual environment
        source venv/bin/activate      # Mac/Linux
        # OR
        venv\\Scripts\\activate       # Windows
        
        # Start the API server
        uvicorn api:app --reload
        ```
        
        ### **Step 2: Wait for API to Start**
        
        You should see messages like:
        ```
        INFO:     Started server process [...]
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://127.0.0.1:8000
        ```
        
        ### **Step 3: Refresh This Page**
        
        Once the API is running, refresh this page and try again.
        
        ---
        
        ### **Quick Test (Optional)**
        
        To test if the API is working, run this in another terminal:
        
        ```bash
        curl http://localhost:8000/health
        ```
        
        You should get a JSON response like:
        ```json
        {"status": "healthy", "model_loaded": true}
        ```
        """)
        
    except requests.exceptions.Timeout:
        st.error("## ⏱️ Request Timeout")
        st.markdown("""
        The request took too long to process. This might be because:
        
        - The API server is busy
        - Your internet connection is slow
        - The model is taking longer than expected
        
        **Try:**
        1. Wait a few seconds and click "Predict Price Now" again
        2. Restart the API server
        3. Check your internet connection
        """)
        
    except Exception as e:
        st.error(f"## 🚨 Unexpected Error: {str(e)}")
        st.code(f"Full error details:\n{str(e)}", language="python")
        st.markdown("""
        **To report this error:**
        1. Take a screenshot of this error message
        2. Check the terminal where API is running for more details
        3. Try restarting both the API and this web interface
        """)
    
    finally:
        # Clear progress bar
        progress_bar.empty()
        status_text.empty()

else:
    # Welcome screen (when button not pressed)
    st.markdown("""
    ## 🎯 Welcome to Your Smart Price Assistant!
    
    This AI-powered tool helps you determine the **fair market price** for any used item. 
    Whether you're selling a phone, laptop, or other electronics, get data-driven price 
    recommendations in seconds.
    
    ### ✨ **How It Works:**
    
    1. **Enter Details** - Fill in your item's information in the sidebar
    2. **AI Analysis** - Our machine learning model analyzes market data
    3. **Get Insights** - Receive price predictions and selling recommendations
    
    ### 📊 **What You'll Get:**
    
    - ✅ **Accurate Price Prediction** - Based on real market data
    - 📈 **Price Range** - Understand the market value spread
    - 💡 **Selling Tips** - Optimize your listing strategy
    - 🏪 **Platform Comparison** - See estimated prices across marketplaces
    
    ### 🚀 **Ready to Start?**
    
    **Simply fill out the form on the left** and click the **blue "PREDICT PRICE NOW" button**!
    
    ---
    
    ### 💎 **Popular Items to Try:**
    
    Use the quick example buttons in the sidebar or enter your own item details.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 1rem; background: #F3F4F6; border-radius: 10px; height: 200px;">
            <h3 style="color: #1E3A8A;">🤖 AI-Powered</h3>
            <p>Machine learning models trained on thousands of sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; background: #F3F4F6; border-radius: 10px; height: 200px;">
            <h3 style="color: #1E3A8A;">📊 Data-Driven</h3>
            <p>Real-time market analysis and price trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; background: #F3F4F6; border-radius: 10px; height: 200px;">
            <h3 style="color: #1E3A8A;">⚡ Instant Results</h3>
            <p>Get price predictions in seconds, not hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tips section
    with st.expander("💡 Pro Tips for Accurate Pricing"):
        st.markdown("""
        **For best results:**
        
        1. **Be specific in the title** - Include brand, model, storage, color
        2. **Accurate condition** - Be honest about wear and tear
        3. **Include shipping costs** - These affect final price
        4. **Mention defects** - Transparency builds trust with buyers
        
        **Example of a good title:** *"Apple iPhone 12 128GB Black - Unlocked - Excellent Condition"*
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>Used Product Price Analyzer v1.0 • Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;"> 
        <a href="http://localhost:8000" target="_blank" style="color: #3B82F6; text-decoration: none;">API Documentation</a> • 
        <a href="http://localhost:8000/docs" target="_blank" style="color: #3B82F6; text-decoration: none;">Swagger UI</a> • 
        <a href="#" style="color: #3B82F6; text-decoration: none;" onclick="alert('To train model: python main.py')">Train Model</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh if API comes online
try:
    requests.get("http://localhost:8000/health", timeout=1)
except:
    pass  # API not running, that's okay