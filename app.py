import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import datetime
import app_task2_inference_snippet as task2

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IFFCO Trade Strategy Engine",
    page_icon="🧊",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

st.title("🚀 IFFCO Intelligent Trade Planner (V7 Enterprise)")

# --- SIDEBAR & LOADING ---
st.sidebar.header("⚙️ System Status")

@st.cache_resource
@st.cache_resource
def load_resources():
    try:
        # ---------- Task 1 ----------
        disc_model = joblib.load(os.path.join(MODEL_DIR, 'task1_discount_regressor_v3.pkl'))
        disc_meta  = joblib.load(os.path.join(MODEL_DIR, 'task1_metadata_v3.pkl'))

        # ---------- Task 2 (Corrected Model) ----------
        # 1. Load Corrected Model Artifacts
        task2_model = joblib.load(os.path.join(MODEL_DIR, "task2_reboot_model.pkl"))
        metadata    = joblib.load(os.path.join(MODEL_DIR, "task2_reboot_metadata.pkl"))
        states      = joblib.load(os.path.join(MODEL_DIR, "task2_reboot_states.pkl"))
        history_df  = joblib.load(os.path.join(MODEL_DIR, "history_monthly.pkl"))

        # 2. Seasonality data
        seasonality_df = pd.read_csv(os.path.join(MODEL_DIR, "product_seasonality_correct.csv"))

        # ---------- Task 3 (CSV Support) ----------
        seg_path_csv = os.path.join(MODEL_DIR, 'customer_segments.csv')
        if os.path.exists(seg_path_csv):
             seg_data = pd.read_csv(seg_path_csv)
             # Ensure required columns exist
             if 'Total_Spend' not in seg_data.columns and 'TOTAL_SALES' in seg_data.columns:
                 seg_data['Total_Spend'] = seg_data['TOTAL_SALES']
             if 'Claim_Count' not in seg_data.columns and 'FREQUENCY' in seg_data.columns:
                 seg_data['Claim_Count'] = seg_data['FREQUENCY']
             if 'Avg_Claim_Size' not in seg_data.columns:
                 seg_data['Avg_Claim_Size'] = seg_data.get('AVG_ORDER_VALUE', 1000)
        else:
             seg_data = None

        return disc_model, disc_meta, task2_model, metadata, states, seasonality_df, history_df, seg_data

    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        # Return 8 Nones to match the unpacking count
        return None, None, None, None, None, None, None, None


# Load Models
discount_model, disc_meta, task2_model, metadata, states, seasonality_df, history_df, segmentation_data = load_resources()

if discount_model and task2_model:
    st.sidebar.success("✅ Models Loaded")
    st.sidebar.caption(f"Task 2 Test APE: {metadata.get('test_median_ape', 0):.1f}%")
else:
    st.sidebar.warning("⚠️ Models missing. Please check file names.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["💰 Task 1: Discount Optimizer", "📈 Task 2: Sales Forecast (V7)", "🧩 Task 3: Strategy Matrix"])


# ==========================================
# TAB 1: DISCOUNT OPTIMIZATION
# ==========================================
with tab1:
    st.subheader("Optimal Discount Recommendation")
    if disc_meta:
        col1, col2, col3 = st.columns(3)
        with col1:
            customer = st.selectbox("Customer Name", disc_meta['customers'])
            global_acc = disc_meta['cust_global_map'].get(customer, "Unknown")
            st.caption(f"Linked Global Account: {global_acc}")
            channel = st.selectbox("Channel", disc_meta['channels'])
        with col2:
            product = st.selectbox("Product Group", disc_meta['products'])
            tactic = st.selectbox("Tactic Type", disc_meta['tactics'])
            ctype = st.selectbox("Contract Type", disc_meta['types'])
        with col3:
            start_date = st.date_input("Contract Start", datetime.date(2026, 1, 1))
            duration = st.slider("Duration (Months)", 1, 12, 12)
        
        if st.button("🤖 Calculate Optimal Discount", type="primary"):
            input_data = {
                'CUSTOMER': [customer], 'GLOBAL_ACCOUNT': [global_acc],
                'CHANNEL': [channel], 'TYPE': [ctype],
                'TACTIC_TYPE': [tactic], 'PRODUCT_GROUP': [product],
                'START_MONTH_NUM': [start_date.month],
                'END_MONTH_NUM': [(start_date.month + duration) % 12 or 12]
            }
            try:
                prediction = discount_model.predict(pd.DataFrame(input_data))[0]
                st.divider()
                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.metric(label="Recommended Discount", value=f"{prediction:.2f}%")
                with r_col2:
                    if prediction > 14.0: st.warning("⚠️ High Discount: Above 14%.")
                    elif prediction < 11.0: st.success("✅ Margin Safe: Strong profitability.")
                    else: st.info("ℹ️ Standard Range.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
# ==========================================
# TAB 2: SALES FORECASTING (CORRECTED MODEL)
# ==========================================
with tab2:
    st.subheader("Sales Forecasting (Corrected Model)")
    st.caption("Uses Contract_Claim_Dump data with actual product categories")
    
    if metadata and task2_model:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # Get customers from states
            available_customers = sorted(set([k[0] for k in states.keys()]))
            selected_customer = st.selectbox("Customer", available_customers)
        
        with c2:
            # Get products for selected customer
            customer_products = sorted(set([k[1] for k in states.keys() if k[0] == selected_customer]))
            selected_product = st.selectbox("Product Group", customer_products if customer_products else metadata['products'])
        
        with c3:
            forecast_months = st.slider("Forecast Period (Months)", 1, 12, 6)
        
        if st.button("📈 Generate Forecast", type="primary"):
            try:
                from datetime import datetime
                import app_task2_inference_corrected as task2_inf
                
                # Load the inference module's model (ensure it uses the same artifacts)
                task2_inf.model = task2_model
                task2_inf.metadata = metadata
                task2_inf.states = states
                task2_inf.df_seasonality = seasonality_df
                
                # Generate forecast
                result = task2_inf.forecast_sales(selected_customer, selected_product, months_ahead=forecast_months)
                
                if result['status'] == 'success':
                    st.success(f"✓ Forecast generated for {selected_customer} - {selected_product}")
                    
                    # Display metrics
                    forecasts = result['forecast']
                    total_forecast = sum([f['forecast'] for f in forecasts])
                    avg_seasonality = np.mean([f['seasonality'] for f in forecasts])
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Forecast", f"${total_forecast:,.0f}")
                    m2.metric("Avg Monthly", f"${total_forecast/len(forecasts):,.0f}")
                    m3.metric("Avg Seasonality", f"{avg_seasonality:.2f}x")
                    
                    # Create forecast table
                    forecast_df = pd.DataFrame(forecasts)
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Historical data (if available)
                    if history_df is not None:
                        hist = history_df[(history_df['CUSTOMER'] == selected_customer) & 
                                         (history_df['PRODUCT_GROUP'] == selected_product)]
                        if not hist.empty:
                            fig.add_trace(go.Scatter(
                                x=hist['Plot_Date'],
                                y=hist['SALES'],
                                mode='lines+markers',
                                name='Historical',
                                line=dict(color='gray', width=2)
                            ))
                    
                    # Forecast data
                    now = datetime.now()
                    forecast_dates = [datetime(now.year if f['month'] > now.month else now.year + 1, f['month'], 1) 
                                     for f in forecasts]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_df['forecast'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#1f77b4', width=3, dash='dot')
                    ))
                    
                    fig.update_layout(
                        title=f"Sales Forecast: {selected_product}",
                        xaxis_title="Date",
                        yaxis_title="Sales (AED)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table
                    st.dataframe(
                        forecast_df[['month_name', 'forecast', 'seasonality', 'confidence']]
                        .rename(columns={
                            'month_name': 'Month',
                            'forecast': 'Forecast (AED)',
                            'seasonality': 'Seasonality',
                            'confidence': 'Confidence'
                        })
                        .style.format({
                            'Forecast (AED)': '${:,.2f}',
                            'Seasonality': '{:.2f}x',
                            'Confidence': '{:.0%}'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.error(result['message'])
                    
            except Exception as e:
                st.error(f"Forecast Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.error("❌ Model not loaded. Please run train_task2_corrected.py first.")
    
# ==========================================
# TAB 3: STRATEGY MATRIX
# ==========================================
with tab3:
    st.subheader("Customer Strategy Matrix")
    if segmentation_data is not None and len(segmentation_data) > 0:
        try:
            plot_df = segmentation_data.reset_index() if hasattr(segmentation_data, 'reset_index') else segmentation_data
            
            # Check if required columns exist
            required_cols = ['Total_Spend', 'Claim_Count', 'Avg_Claim_Size', 'Segment_Name', 'CUSTOMER_NAME']
            missing_cols = [col for col in required_cols if col not in plot_df.columns]
            
            if missing_cols:
                st.warning(f"Segmentation data missing columns: {missing_cols}")
                st.info("Available columns: " + ", ".join(plot_df.columns.tolist()[:10]))
                st.caption("Please run task3_segmentation.py to generate proper segmentation data")
            else:
                valid_plot_df = plot_df[plot_df['Total_Spend'] > 1].copy()
                valid_plot_df['Bubble_Size'] = valid_plot_df['Avg_Claim_Size'].abs().fillna(1)
                
                search_cust = st.multiselect("Highlight Specific Customers:", valid_plot_df['CUSTOMER_NAME'].unique())
                
                if search_cust:
                    valid_plot_df['Color'] = valid_plot_df['CUSTOMER_NAME'].apply(lambda x: "Selected" if x in search_cust else "Others")
                    color_map = {"Selected": "red", "Others": "lightgrey"}
                    fig = px.scatter(valid_plot_df, x="Claim_Count", y="Total_Spend",
                                     size="Bubble_Size", color="Color", color_discrete_map=color_map,
                                     hover_name="CUSTOMER_NAME", log_x=True, log_y=True, 
                                     title="Investment vs. Frequency")
                else:
                    fig = px.scatter(valid_plot_df, x="Claim_Count", y="Total_Spend",
                                     size="Bubble_Size", color="Segment_Name",
                                     hover_name="CUSTOMER_NAME", log_x=True, log_y=True,
                                     title="Investment vs. Frequency")

                st.plotly_chart(fig, use_container_width=True)
                sel_seg = st.selectbox("Filter Segment", plot_df['Segment_Name'].unique())
                st.dataframe(plot_df[plot_df['Segment_Name'] == sel_seg][['CUSTOMER_NAME', 'Total_Spend', 'Claim_Count']], use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying segmentation data: {e}")
            st.caption("Please run task3_segmentation.py to generate proper customer segments")
    else:
        st.warning("Segmentation data not available.")
        st.info("Run task3_segmentation.py to generate customer segmentation analysis")