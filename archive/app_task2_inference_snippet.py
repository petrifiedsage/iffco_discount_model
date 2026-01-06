"""
STREAMLIT APP - TASK 2 INFERENCE SNIPPET
Add this to your app.py in Tab 2 for sales forecasting.

This snippet loads the trained model and performs recursive forecasting
for any customer-product combination.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# LOAD MODEL ARTIFACTS (Cache for performance)
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load model, states, metadata, and seasonality."""
    try:
        MODEL_DIR = Path("models")
        model = joblib.load(MODEL_DIR / "task2_reboot_model.pkl")
        states = joblib.load(MODEL_DIR / "task2_reboot_states.pkl")
        metadata = joblib.load(MODEL_DIR / "task2_reboot_metadata.pkl")
        
        # Load seasonality (enhanced multi-year version first, then fallback)
        seasonality_path = MODEL_DIR / "product_seasonality_enhanced.csv"
        if not seasonality_path.exists():
            seasonality_path = MODEL_DIR / "product_seasonality_claims.csv"
        if not seasonality_path.exists():
            seasonality_path = MODEL_DIR / "product_seasonality.csv"
        
        if seasonality_path.exists():
            df_season = pd.read_csv(seasonality_path)
            # Create lookup dictionaries for seasonality and confidence
            seasonality_dict = {}
            confidence_dict = {}
            for _, row in df_season.iterrows():
                seasonality_dict[(row['PRODUCT_GROUP'], int(row['MONTH']))] = row['SEASONALITY_INDEX']
                # Use confidence if available (enhanced file), else default to 1.0
                confidence = row['CONFIDENCE'] if 'CONFIDENCE' in df_season.columns else 1.0
                confidence_dict[(row['PRODUCT_GROUP'], int(row['MONTH']))] = confidence
        else:
            seasonality_dict = {}
            confidence_dict = {}
        
        return model, states, metadata, seasonality_dict
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, {}


# ============================================================================
# RECURSIVE FORECASTING FUNCTION
# ============================================================================
def forecast_recursive(model, feature_cols, initial_state, product, seasonality_dict, months_ahead=12, confidence_dict=None):
    """
    Perform recursive multi-step forecasting with enhanced features and product-specific seasonality.
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        initial_state: Dictionary with initial lag values
        product: Product name for seasonality lookup
        seasonality_dict: Dictionary mapping (product, month) to seasonality index
        months_ahead: Number of months to forecast
        confidence_dict: Optional dictionary mapping (product, month) to confidence weights
        
    Returns:
        List of forecast dictionaries
    """
    forecasts = []
    
    # Copy initial state to avoid modifying original
    current_state = initial_state.copy()
    current_date = pd.to_datetime(current_state['last_date'])
    
    # Store recent sales for rolling averages (last 12 values)
    sales_history = [h['SALES'] for h in current_state['recent_history'][-12:]]
    
    # Calculate baseline averages from state
    customer_avg = current_state.get('customer_avg', np.mean(sales_history) if sales_history else 1000)
    product_avg = current_state.get('product_avg', np.mean(sales_history) if sales_history else 1000)
    customer_product_avg = current_state.get('customer_product_avg', np.mean(sales_history) if sales_history else 1000)
    
    # Calculate historical baseline for bounds checking
    historical_mean = np.mean(sales_history) if sales_history else customer_product_avg
    historical_std = np.std(sales_history) if len(sales_history) > 1 else historical_mean * 0.3
    
    if confidence_dict is None:
        confidence_dict = {}
    
    for month_idx in range(1, months_ahead + 1):
        # Move to next month
        next_date = current_date + relativedelta(months=month_idx)
        
        # Get seasonality index and confidence for this product and month
        seasonality_index = seasonality_dict.get((product, next_date.month), 1.0)
        seasonality_confidence = confidence_dict.get((product, next_date.month), 1.0)
        
        # Calculate trend from recent history
        if len(sales_history) >= 3:
            trend_3m = (sales_history[-1] - sales_history[-3]) / 3 if len(sales_history) >= 3 else 0
            lag_3_std = np.std(sales_history[-3:])
            momentum = sales_history[-1] / (np.mean(sales_history[-6:]) + 1) if len(sales_history) >= 6 else 1.0
        else:
            trend_3m = 0
            lag_3_std = 0
            momentum = 1.0
        
        # Growth rate
        if len(sales_history) >= 12:
            growth_rate = (sales_history[-1] - sales_history[-12]) / (sales_history[-12] + 1)
            growth_rate = np.clip(growth_rate, -5, 5)
        else:
            growth_rate = 0
        
        # Product volume category (estimate from average)
        if customer_product_avg > product_avg * 1.5:
            product_volume_cat = 2
        elif customer_product_avg < product_avg * 0.5:
            product_volume_cat = 0
        else:
            product_volume_cat = 1
        
        # Build feature vector (must match training feature order)
        features = {
            'LAG_1': current_state['lag_1'],
            'LAG_2': current_state['lag_2'],
            'LAG_3': current_state['lag_3'],
            'LAG_12': current_state['lag_12'],
            'LAG_3_AVG': current_state['lag_3_avg'],
            'LAG_6_AVG': current_state['lag_6_avg'],
            'LAG_3_STD': lag_3_std,
            'CUSTOMER_AVG': customer_avg,
            'PRODUCT_AVG': product_avg,
            'CUSTOMER_PRODUCT_AVG': customer_product_avg,
            'RELATIVE_IMPORTANCE': customer_product_avg / (customer_avg + 1),
            'TREND_3M': trend_3m,
            'MOMENTUM': momentum,
            'GROWTH_RATE': growth_rate,
            'MONTH': next_date.month,
            'QUARTER': next_date.quarter,
            'YEAR': next_date.year,
            'IS_SUMMER': 1 if next_date.month in [6, 7, 8] else 0,
            'IS_RAMADAN_PERIOD': 1 if next_date.month in [3, 4] else 0,
            'SEASONALITY_INDEX': seasonality_index,
            'PRODUCT_VOLUME_CAT': product_volume_cat
        }
        
        # Create feature array in correct order
        X = np.array([[features[col] for col in feature_cols]])
        
        # Predict (in log space)
        y_pred_log = model.predict(X)[0]
        y_pred_raw = np.expm1(y_pred_log)  # Convert back from log
        y_pred_raw = max(0, y_pred_raw)  # Ensure non-negative
        
        # Apply PRODUCT-SPECIFIC seasonality with confidence weighting
        # Blend between seasonality index and baseline (1.0) based on confidence
        # High confidence (1.0) = fully apply seasonality
        # Low confidence (0.5) = blend 50% with baseline
        adjusted_seasonality = 1.0 + (seasonality_index - 1.0) * seasonality_confidence
        y_pred_seasonal = y_pred_raw * adjusted_seasonality
        
        # Apply MODERATE volume-based boost as secondary adjustment
        # This addresses the fact that model is conservative for sparse customer-products
        if customer_product_avg < 200:  # Very sparse
            volume_boost = 1.8  # Moderate boost (was 2.0, reduced for extreme seasonality products)
        elif customer_product_avg < 1000:  # Sparse
            volume_boost = 1.4  # Moderate boost (was 1.6)
        elif customer_product_avg < 5000:  # Normal volume
            volume_boost = 1.2  # Slight boost (was 1.3)
        else:  # High volume - model is accurate
            volume_boost = 1.05  # Very slight boost (was 1.1)
        
        y_pred = y_pred_seasonal * volume_boost
        
        # Bound predictions to wider reasonable range (±5σ for more variance)
        lower_bound = max(0, historical_mean - 5 * historical_std)
        upper_bound = historical_mean + 5 * historical_std
        y_pred = np.clip(y_pred, lower_bound, upper_bound)
        
        # Store forecast
        forecasts.append({
            'date': next_date,
            'forecast': y_pred,
            'month': next_date.strftime('%Y-%m')
        })
        
        # Update state for next iteration (shift lags)
        sales_history.append(y_pred)  # Add new prediction to history
        
        current_state['lag_12'] = sales_history[-12] if len(sales_history) >= 12 else 0
        current_state['lag_3'] = current_state['lag_2']
        current_state['lag_2'] = current_state['lag_1']
        current_state['lag_1'] = y_pred
        
        # Update rolling averages
        recent_3 = sales_history[-3:] if len(sales_history) >= 3 else sales_history
        recent_6 = sales_history[-6:] if len(sales_history) >= 6 else sales_history
        current_state['lag_3_avg'] = np.mean(recent_3) if recent_3 else 0
        current_state['lag_6_avg'] = np.mean(recent_6) if recent_6 else 0
    
    return forecasts


# ============================================================================
# STREAMLIT UI - TAB 2: SALES FORECASTING
# ============================================================================
def render_sales_forecast_tab():
    """
    Main function to render Tab 2 in your Streamlit app.
    Call this function inside your Tab 2.
    """
    st.header("📈 Task 2: Sales Forecasting")
    st.markdown("**12-Month Recursive Forecast with Historical Context**")
    
    # Load artifacts
    model, states, metadata, seasonality_dict = load_model_artifacts()
    
    if model is None:
        st.error("❌ Model not loaded. Please run `train_task2_reboot.py` first.")
        return
    
    st.success(f"✅ Model loaded (Trained: {metadata['trained_date']})")
    st.info(f"📊 Data covers {metadata['n_customers']} customers, {metadata['n_products']} products | Seasonality: {len(seasonality_dict)} patterns")
    
    # ========================================================================
    # USER INPUTS
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        selected_customer = st.selectbox(
            "Select Customer",
            options=metadata['customers'],
            index=0
        )
    
    with col2:
        # Filter products that exist for this customer
        available_products = [
            prod for (cust, prod) in states.keys() if cust == selected_customer
        ]
        
        if not available_products:
            st.warning("No products found for this customer")
            return
        
        selected_product = st.selectbox(
            "Select Product",
            options=available_products,
            index=0
        )
    
    months_to_forecast = st.slider("Forecast Horizon (Months)", 3, 24, 12)
    
    # ========================================================================
    # FORECAST BUTTON
    # ========================================================================
    if st.button("🔮 Generate Forecast", type="primary"):
        key = (selected_customer, selected_product)
        
        if key not in states:
            st.error("❌ No historical data found for this combination")
            return
        
        with st.spinner("Generating forecast..."):
            # Get initial state
            initial_state = states[key]
            
            # Perform recursive forecast
            forecasts = forecast_recursive(
                model=model,
                feature_cols=metadata['feature_cols'],
                initial_state=initial_state,
                product=selected_product,
                seasonality_dict=seasonality_dict,
                months_ahead=months_to_forecast,
                confidence_dict=confidence_dict
            )
            
            # ================================================================
            # PREPARE DATA FOR VISUALIZATION
            # ================================================================
            # Historical data (last 24 months)
            hist_data = initial_state['recent_history'][-24:]
            df_hist = pd.DataFrame(hist_data)
            df_hist['MONTH_START'] = pd.to_datetime(df_hist['MONTH_START'])
            df_hist = df_hist.sort_values('MONTH_START')
            
            # Forecast data
            df_forecast = pd.DataFrame(forecasts)
            
            # ================================================================
            # PLOTLY CHART
            # ================================================================
            fig = go.Figure()
            
            # Historical line
            fig.add_trace(go.Scatter(
                x=df_hist['MONTH_START'],
                y=df_hist['SALES'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=df_forecast['date'],
                y=df_forecast['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            # Styling
            fig.update_layout(
                title=f"Sales Forecast: {selected_customer} - {selected_product}",
                xaxis_title="Month",
                yaxis_title="Sales Amount ($)",
                hovermode='x unified',
                height=500,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ================================================================
            # SUMMARY METRICS
            # ================================================================
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_forecast = sum(f['forecast'] for f in forecasts)
            avg_forecast = total_forecast / len(forecasts)
            last_historical = df_hist['SALES'].iloc[-1]
            growth_rate = ((avg_forecast - last_historical) / last_historical * 100) if last_historical > 0 else 0
            
            with col1:
                st.metric(
                    "Total Forecast",
                    f"${total_forecast:,.0f}",
                    delta=f"{months_to_forecast} months"
                )
            
            with col2:
                st.metric(
                    "Avg Monthly",
                    f"${avg_forecast:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Last Historical",
                    f"${last_historical:,.0f}"
                )
            
            with col4:
                st.metric(
                    "Trend",
                    f"{growth_rate:+.1f}%",
                    delta="vs Last Month"
                )
            
            # ================================================================
            # DETAILED FORECAST TABLE
            # ================================================================
            st.subheader("📅 Detailed Forecast")
            
            df_display = df_forecast.copy()
            df_display['forecast'] = df_display['forecast'].apply(lambda x: f"${x:,.2f}")
            df_display = df_display.rename(columns={
                'month': 'Month',
                'forecast': 'Forecasted Sales'
            })
            
            st.dataframe(
                df_display[['Month', 'Forecasted Sales']],
                use_container_width=True,
                hide_index=True
            )
            
            # ================================================================
            # DOWNLOAD BUTTON
            # ================================================================
            csv = df_forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv,
                file_name=f"forecast_{selected_customer}_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# ============================================================================
# HOW TO USE IN YOUR APP.PY
# ============================================================================
"""
In your app.py, add this to Tab 2:

```python
import app_task2_inference_snippet as task2

# Inside your tab2 code:
with tab2:
    task2.render_sales_forecast_tab()
```

OR, if you want to inline it, just copy the render_sales_forecast_tab() function
and the helper functions above directly into your app.py.
"""
