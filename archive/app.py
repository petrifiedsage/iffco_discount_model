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

        # ---------- Task 2 (V7 Complex Loading) ----------
        # 1. Load Encoders & Meta
        encoders   = joblib.load(os.path.join(MODEL_DIR, "task2_encoders_v7.pkl"))
        metadata   = joblib.load(os.path.join(MODEL_DIR, "task2_metadata_v7.pkl"))
        history_df = joblib.load(os.path.join(MODEL_DIR, "history_monthly.pkl"))

        # 2. Load Segment Models (Handle Dictionary Structure)
        seg_models_path = os.path.join(MODEL_DIR, "segment_models_v7.pkl")
        segment_models_raw = joblib.load(seg_models_path) if os.path.exists(seg_models_path) else {}
        
        # Process Segment Models into standard format
        segment_models = {}
        for seg, models in segment_models_raw.items():
            if models.get("lgb"):
                segment_models[seg] = {"model": models["lgb"], "type": "raw"} # LightGBM = Raw Sales
            else:
                segment_models[seg] = {"model": models["hgb"], "type": "log"} # HGB = Log Sales

        # 3. Load Global Model (Fallback)
        global_model = {"model": None, "type": "log"}
        if os.path.exists(os.path.join(MODEL_DIR, "task2_sales_forecasting_model_v7_lgb.pkl")):
             global_model = {"model": joblib.load(os.path.join(MODEL_DIR, "task2_sales_forecasting_model_v7_lgb.pkl")), "type": "raw"}
        elif os.path.exists(os.path.join(MODEL_DIR, "task2_sales_forecasting_model_v7_hgb.pkl")):
             global_model = {"model": joblib.load(os.path.join(MODEL_DIR, "task2_sales_forecasting_model_v7_hgb.pkl")), "type": "log"}

        # ---------- Task 3 (CSV Support) ----------
        # Training script saves CSV now, not PKL
        seg_path_csv = os.path.join(MODEL_DIR, 'customer_segments.csv')
        if os.path.exists(seg_path_csv):
             seg_data = pd.read_csv(seg_path_csv)
        else:
             seg_data = None

        return disc_model, disc_meta, global_model, encoders, metadata, history_df, seg_data, segment_models

    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        # Return 8 Nones to match the unpacking count
        return None, None, None, None, None, None, None, None


# Load Models
discount_model, disc_meta, global_model, encoders, metadata, history_df, segmentation_data, segment_models = load_resources()



if discount_model and global_model:
    st.sidebar.success("✅ Models Loaded")
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
"""
# ==========================================
# TAB 1: DISCOUNT OPTIMIZATION (LEAK-FREE)
# ==========================================
with tab1:
    st.subheader("Optimal Discount Recommendation (AI Assisted)")

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
            contract_year = st.selectbox(
                "Contract Year",
                [2024, 2025, 2026],
                index=0
            )
            st.caption("Prediction based on historical contracts (2018–2023)")

        if st.button("🤖 Calculate Optimal Discount", type="primary"):
            try:
                # 🔒 LEAK-FREE INPUT (matches training features exactly)
                input_df = pd.DataFrame([{
                    'CUSTOMER': customer,
                    'GLOBAL_ACCOUNT': global_acc,
                    'CHANNEL': channel,
                    'TYPE': ctype,
                    'TACTIC_TYPE': tactic,
                    'PRODUCT_GROUP': product,
                    'YEAR_CREATED': contract_year
                }])

                prediction = discount_model.predict(input_df)[0]

                st.divider()
                r_col1, r_col2 = st.columns([1, 2])

                with r_col1:
                    st.metric(
                        label="Recommended Rebate %",
                        value=f"{prediction:.2f}%"
                    )

                with r_col2:
                    if prediction >= 14:
                        st.warning("⚠️ High Discount — Margin Risk")
                    elif prediction <= 10:
                        st.success("✅ Margin Safe Zone")
                    else:
                        st.info("ℹ️ Standard Negotiation Range")

                st.caption(
                    "Model validated on future contracts only • Leak-free • MAE ≈ 2–3%"
                )

            except Exception as e:
                st.error(f"Prediction Error: {e}")
"""
# ==========================================
# TAB 2: SALES FORECASTING (V7 – ROBUST)
# ==========================================
with tab2:
    task2.render_sales_forecast_tab()
    """
    st.subheader("Future Demand Forecasting (Segment-Aware AI)")
    st.caption("Seasonality and trends are learned automatically from historical behavior.")

    # 1. INPUTS
    c1, c2, c3 = st.columns(3)
    with c1:
        cust = st.selectbox("Customer", metadata["customers"])
        # Region auto-lookup
        region_default = metadata["regions"][0] if metadata["regions"] else "United Arab Emirates"
        region = st.selectbox("Region", metadata["regions"], index=0)
    with c2:
        prod = st.selectbox("Product Group", metadata["products"])
    with c3:
        # Dynamic Next 12 Months Generator
        today = datetime.datetime.today()
        next_12_months = []
        for i in range(1, 13):
             next_month = today.month + i
             year_offset = (next_month - 1) // 12
             month_val = (next_month - 1) % 12 + 1
             next_12_months.append((today.year + year_offset, month_val))
        
        month_opts = [f"{y}-{m:02d}" for y, m in next_12_months]
        selected_periods = st.multiselect("Forecast Period", month_opts, default=month_opts)

    # 2. GENERATE FORECAST
    if st.button("📈 Generate Forecast"):
        try:
            # A. Prepare Encoders
            cust_avg = encoders["cust_means"].get(cust, np.mean(list(encoders["cust_means"].values())))
            prod_avg = encoders["prod_means"].get(prod, np.mean(list(encoders["prod_means"].values())))
            geo_avg  = encoders["geo_means"].get(region, np.mean(list(encoders["geo_means"].values())))
            cp_avg   = encoders["cust_prod_means"].get((cust, prod), prod_avg)
            recent   = encoders["recent_means"].get((cust, prod), cp_avg)

            # B. Determine Segment & Model
            if cust_avg > prod_avg * 1.5:
                segment = "Key_Accounts"
            elif cust_avg < prod_avg * 0.6:
                segment = "Low_Volume"
            else:
                segment = "Mid_Market"
            
            selected = segment_models.get(segment, global_model)
            model = selected["model"]
            is_log_model = (selected["type"] == "log")

            # C. Prepare History (For Lags)
            if history_df is not None:
                hist = history_df[(history_df["CUSTOMER"] == cust) & (history_df["PRODUCT_GROUP"] == prod)]
                hist = hist.sort_values("Plot_Date")
                series = list(hist["SALES"].values)
            else:
                series = []

            # Seed with 'recent' if history is empty
            if len(series) < 3: series = [recent] * 3

            # D. Recursive Prediction Loop
            forecasts = []
            
            for per in selected_periods:
                y_str, m_str = per.split('-')
                year_int = int(y_str)
                month_int = int(m_str)
                is_summer = 1 if month_int in [6, 7, 8] else 0

                # Calculate Lags from the growing 'series' list
                lag1 = series[-1]
                lag2 = series[-2]
                lag3 = series[-3]
                roll_mean = np.mean(series[-3:])
                roll_std  = np.std(series[-3:])

                # Promo lookups
                season_dict = encoders.get("promo_seasonality", {}).get("seasonality", {})
                freq_dict = encoders.get("promo_seasonality", {}).get("freq", {})
                sum_dict = encoders.get("promo_seasonality", {}).get("sum", {})
                
                promo_key = (prod, month_int)
                promo = season_dict.get(promo_key, 1.0)
                pf    = freq_dict.get(promo_key, 0)
                ps    = sum_dict.get(promo_key, 0)

                # Feature Vector (22 Cols - Matches Training Exactly)
                X = pd.DataFrame([[
                    cust_avg, prod_avg, geo_avg, cp_avg, recent,
                    lag1, lag2, lag3, roll_mean, roll_std,
                    month_int, is_summer, promo, pf, ps,
                    ps / (prod_avg + 1e-6), cp_avg / (prod_avg + 1e-6), recent * promo,
                    0, 0, 0, 0 # Trend/Pct assumed stable
                ]], columns=[
                    "Cust_Avg", "Prod_Avg", "Geo_Avg", "Cust_Prod_Avg", "RECENT_AVG",
                    "SALES_LAG_1", "SALES_LAG_2", "SALES_LAG_3", "SALES_ROLL_MEAN_3", "SALES_ROLL_STD_3",
                    "MONTH", "Is_Summer", "PROMO_SEASONALITY_INDEX", "PROMO_FREQ", "PROMO_SUM",
                    "PROMO_SUM_OVER_PROD", "CUST_PROD_OVER_PROD", "RECENT_X_PROMO",
                    "TREND_SLOPE_3", "TREND_SLOPE_6", "PCT_CHANGE_LAG1", "PCT_CHANGE_LAG3"
                ])

                # Predict
                raw_pred = model.predict(X)[0]
                
                # Apply Scale (Log vs Raw)
                pred = max(0, np.expm1(raw_pred)) if is_log_model else max(0, raw_pred)

                series.append(pred) # Feed prediction back for next month
                forecasts.append({"Date": pd.Timestamp(f"{year_int}-{month_int}-01"), "Value": pred, "Type": "Forecast"})

            # E. Visualize
            res_df = pd.DataFrame(forecasts)
            
            if not history_df.empty:
                hist_plot = hist.rename(columns={"Plot_Date": "Date", "SALES": "Value"})
                hist_plot["Type"] = "Historical"
                plot_df = pd.concat([hist_plot.tail(24)[["Date", "Value", "Type"]], res_df])
            else:
                plot_df = res_df

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Projected Volume", f"{res_df['Value'].sum():,.0f} AED")
            m2.metric("Customer Segment", segment)
            m3.metric("Model Logic", "Log-Scale" if is_log_model else "Raw Tweedie")

            fig = px.line(plot_df, x="Date", y="Value", color="Type", markers=True,
                          title=f"Sales Forecast: {prod} ({segment})",
                          color_discrete_map={"Historical": "gray", "Forecast": "#1f77b4"})
            fig.update_traces(patch={"line": {"dash": "dot", "width": 3}}, selector={"legendgroup": "Forecast"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            display_table = res_df.copy()
            display_table["Month"] = display_table["Date"].dt.strftime('%Y-%b')
            st.dataframe(display_table[["Month", "Value"]].style.format({"Value": "{:,.2f}"}), hide_index=True)

        except Exception as e:
            st.error(f"Prediction Failed: {e}")
"""
# ==========================================
# TAB 3: STRATEGY MATRIX
# ==========================================
with tab3:
    st.subheader("Customer Strategy Matrix")
    if segmentation_data is not None:
        plot_df = segmentation_data.reset_index()
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
    else:
        st.warning("Segmentation data missing.")