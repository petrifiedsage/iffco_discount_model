import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import datetime
import traceback

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IFFCO Trade Strategy Engine",
    page_icon="🧊",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

st.title("IFFCO Intelligent Trade Planner")

# --- SIDEBAR & LOADING ---
st.sidebar.header("⚙️ System Status")

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

        # ---------- Task 2 (Ensemble outputs) ----------
        ensemble_predictions = None
        ensemble_path = os.path.join(os.path.dirname(__file__), 'output', 'task2_improved_v4_ensemble_predictions.csv')
        if os.path.exists(ensemble_path):
            try:
                ensemble_predictions = pd.read_csv(ensemble_path)
            except Exception as read_err:
                st.sidebar.warning(f"Could not read ensemble outputs: {read_err}")

        # ---------- Task 2 (2025 Validation & 2026 Retrained Forecast) ----------
        validation_2025 = None
        validation_path = os.path.join(os.path.dirname(__file__), 'output', 'task2_2025_validation_retrained.csv')
        if os.path.exists(validation_path):
            try:
                validation_2025 = pd.read_csv(validation_path)
            except Exception as read_err:
                st.sidebar.warning(f"Could not read 2025 validation: {read_err}")

        forecast_2026 = None
        forecast_path = os.path.join(os.path.dirname(__file__), 'output', 'task2_2026_forecast_retrained.csv')
        if os.path.exists(forecast_path):
            try:
                forecast_2026 = pd.read_csv(forecast_path)
            except Exception as read_err:
                st.sidebar.warning(f"Could not read 2026 forecast: {read_err}")

        return disc_model, disc_meta, task2_model, metadata, states, seasonality_df, history_df, seg_data, ensemble_predictions, validation_2025, forecast_2026

    except Exception as e:
        st.sidebar.error(f"Model Load Error: {e}")
        # Return 9 Nones to match the unpacking count
        return None, None, None, None, None, None, None, None, None


# Load Models
discount_model, disc_meta, task2_model, metadata, states, seasonality_df, history_df, segmentation_data, ensemble_predictions, validation_2025, forecast_2026 = load_resources()

if discount_model and task2_model:
    st.sidebar.success("✅ Models Loaded")
    st.sidebar.caption(f"Task 2 Test APE: {metadata.get('test_median_ape', 0):.1f}%")
else:
    st.sidebar.warning("⚠️ Models missing. Please check file names.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["💰 Task 1: Discount Optimizer", "📈 Task 2: Sales Forecast", "🧩 Task 3: TACTIC Recommendation"])


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
    tab_forecast, tab_val2025, tab_forecast2026 = st.tabs(["On-Demand Forecast", "2025 Validation", "2026 Forecast"])

    with tab_forecast:
        st.subheader("📊 On-Demand Sales Forecasting")
        st.caption("Generate sales forecasts per customer + product using the corrected model.")
        if metadata and task2_model:
            c1, c2, c3 = st.columns(3)

            with c1:
                available_customers = sorted(set([k[0] for k in states.keys()]))
                cust_search = st.text_input("Search Customer")
                cust_options = [c for c in available_customers if (cust_search.strip().lower() in c.lower())] if cust_search else available_customers
                selected_customer = st.selectbox("Customer", cust_options)

            with c2:
                customer_products = sorted(set([k[1] for k in states.keys() if k[0] == selected_customer]))
                prod_search = st.text_input("Search Product")
                prod_options = [p for p in customer_products if (prod_search.strip().lower() in p.lower())] if prod_search else customer_products
                selected_product = st.selectbox("Product Group", prod_options if prod_options else (customer_products if customer_products else metadata['products']))

            with c3:
                forecast_months = st.slider("Forecast Period (Months)", 1, 12, 6)

            if st.button("📈 Generate Forecast", type="primary"):
                try:
                    from datetime import datetime
                    import sys
                    sys.path.insert(0, os.path.dirname(__file__))
                    import app_task2_inference_corrected as task2_inf

                    state_key = (selected_customer, selected_product)
                    if state_key not in states:
                        st.error(f"❌ No historical data for {selected_customer} + {selected_product}")
                        st.info(f"Available combinations: {len(states)}")
                    else:
                        task2_inf.model = task2_model
                        task2_inf.metadata = metadata
                        task2_inf.states = states
                        task2_inf.df_seasonality = seasonality_df

                        result = task2_inf.forecast_sales(selected_customer, selected_product, months_ahead=forecast_months)

                        if result['status'] == 'success':
                            st.success(f"✓ Forecast generated for {selected_customer} - {selected_product}")

                            forecasts = result['forecast']
                            total_forecast = sum([f['forecast'] for f in forecasts])
                            avg_seasonality = np.mean([f['seasonality'] for f in forecasts])

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Total Forecast", f"AED {total_forecast:,.0f}")
                            m2.metric("Avg Monthly", f"AED {total_forecast/len(forecasts):,.0f}")
                            m3.metric("Avg Seasonality", f"{avg_seasonality:.2f}x")

                            forecast_df = pd.DataFrame(forecasts)

                            fig = go.Figure()

                            if history_df is not None:
                                hist = history_df[(history_df['CUSTOMER'] == selected_customer) & (history_df['PRODUCT_GROUP'] == selected_product)]
                                if not hist.empty:
                                    fig.add_trace(go.Scatter(x=hist['Plot_Date'], y=hist['SALES'], mode='lines+markers', name='Historical', line=dict(color='gray', width=2)))

                            now = datetime.now()
                            forecast_dates = [datetime(now.year if f['month'] > now.month else now.year + 1, f['month'], 1) for f in forecasts]

                            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_df['forecast'], mode='lines+markers', name='Forecast', line=dict(color='#1f77b4', width=3, dash='dot')))
                            fig.update_layout(title=f"Sales Forecast: {selected_product}", xaxis_title="Date", yaxis_title="Sales (AED)", hovermode='x unified')
                            st.plotly_chart(fig, width='stretch')

                            st.dataframe(
                                forecast_df[['month_name', 'forecast', 'seasonality', 'confidence']]
                                .rename(columns={'month_name': 'Month', 'forecast': 'Forecast (AED)', 'seasonality': 'Seasonality', 'confidence': 'Confidence'})
                                .style.format({'Forecast (AED)': 'AED {:,.2f}', 'Seasonality': '{:.2f}x', 'Confidence': '{:.0%}'}),
                                hide_index=True,
                                width='stretch'
                            )
                        else:
                            st.error(result['message'])
                except Exception as e:
                    st.error(f"Forecast Error: {e}")
                    st.code(traceback.format_exc())
        else:
            st.error("❌ Model not loaded. Please run train_task2_corrected.py first.")

    with tab_val2025:
        st.subheader("📊 2025 Validation Results (Predicted vs Actual)")
        st.caption("Model validation on 2025 data - Retrained on 2022-2024")

        if validation_2025 is not None and len(validation_2025) > 0:
            val_df = validation_2025.copy()

            val_df['TOTAL_SALES'] = pd.to_numeric(val_df['TOTAL_SALES'], errors='coerce')
            val_df['PREDICTED_SALES'] = pd.to_numeric(val_df['PREDICTED_SALES'], errors='coerce')
            val_df['APE'] = pd.to_numeric(val_df['APE'], errors='coerce')

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Median APE", f"{val_df['APE'].median():.2f}%")
            with m2:
                st.metric("Mean APE", f"{val_df['APE'].mean():.2f}%")
            with m3:
                st.metric("Actual Total (2025)", f"AED {val_df['TOTAL_SALES'].sum():,.0f}")
            with m4:
                st.metric("Predicted Total (2025)", f"AED {val_df['PREDICTED_SALES'].sum():,.0f}")

            col1, col2 = st.columns(2)
            with col1:
                products = ["All"] + sorted(val_df['PRODUCT_GROUP'].unique())
                selected_product_val = st.selectbox("Filter by Product", products)
            with col2:
                seasons = ["All"] + sorted(val_df['SEASON'].unique())
                selected_season_val = st.selectbox("Filter by Season", seasons)

            filtered_val = val_df.copy()
            if selected_product_val != "All":
                filtered_val = filtered_val[filtered_val['PRODUCT_GROUP'] == selected_product_val]
            if selected_season_val != "All":
                filtered_val = filtered_val[filtered_val['SEASON'] == selected_season_val]

            display_val = filtered_val[['PRODUCT_GROUP', 'SEASON', 'TOTAL_SALES', 'PREDICTED_SALES', 'APE']].copy()
            display_val['TOTAL_SALES'] = display_val['TOTAL_SALES'].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "N/A")
            display_val['PREDICTED_SALES'] = display_val['PREDICTED_SALES'].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "N/A")
            display_val['APE'] = display_val['APE'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

            st.dataframe(display_val, hide_index=True, width='stretch')

            st.download_button(label="📥 Download 2025 Validation Results", data=validation_2025.to_csv(index=False), file_name="task2_2025_validation_retrained.csv", mime="text/csv")

            if selected_product_val != "All" or selected_season_val != "All":
                chart_data = filtered_val.sort_values(['PRODUCT_GROUP', 'SEASON'])
                chart_data['Label'] = chart_data['PRODUCT_GROUP'] + ' - ' + chart_data['SEASON']

                fig = go.Figure(data=[
                    go.Bar(name='Actual', x=chart_data['Label'], y=chart_data['TOTAL_SALES'], marker_color='lightblue'),
                    go.Bar(name='Predicted', x=chart_data['Label'], y=chart_data['PREDICTED_SALES'], marker_color='orange')
                ])
                fig.update_layout(title="2025: Actual vs Predicted Sales", xaxis_title="Product - Season", yaxis_title="Sales (AED)", barmode='group', height=400)
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("Run scripts/task2_retrain_2024_validate_2025.py to generate 2025 validation results.")

    with tab_forecast2026:
        st.subheader("📊 2026 Sales Forecast (Retrained Model)")
        st.caption("Forecast based on 2022-2024 training data validated on 2025 - Median APE: 12.35%")

        if forecast_2026 is not None and len(forecast_2026) > 0:
            forecast_sorted = forecast_2026.copy()

            numeric_cols = ['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']
            for col in numeric_cols:
                if col in forecast_sorted.columns:
                    forecast_sorted[col] = pd.to_numeric(forecast_sorted[col], errors='coerce')

            if 'TOTAL_2026' in forecast_sorted.columns:
                forecast_sorted = forecast_sorted.sort_values('TOTAL_2026', ascending=False)

            m1, m2, m3 = st.columns(3)
            with m1:
                total_2026 = forecast_2026[numeric_cols].sum().sum() if numeric_cols[0] in forecast_2026.columns else 0
                st.metric("Total 2026 Forecast", f"AED {total_2026:,.0f}")
            with m2:
                st.metric("Products Forecasted", f"{len(forecast_2026)}")
            with m3:
                st.metric("Seasons Covered", "5 (Winter, Spring, Summer, Fall, Festival)")

            st.write("**Forecast by Product and Season (All Products):**")
            display_full = forecast_sorted[['PRODUCT_GROUP'] + [c for c in numeric_cols if c in forecast_sorted.columns]].copy()
            for col in numeric_cols:
                if col in display_full.columns:
                    display_full[col] = display_full[col].apply(lambda x: f"AED {x:,.0f}" if pd.notna(x) else "N/A")

            st.dataframe(display_full, hide_index=True, width='stretch')

            st.download_button(label="📥 Download 2026 Forecast", data=forecast_2026.to_csv(index=False), file_name="task2_2026_forecast_retrained.csv", mime="text/csv")
        else:
            st.info("Run scripts/task2_retrain_2024_validate_2025.py to generate 2026 forecast.")
    
# ==========================================
# TAB 3: STRATEGY MATRIX
# ==========================================
with tab3:
    st.subheader("Best Tactic per Product (Task 3)")
    try:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        # Prefer model-ranked outputs, fallback to heuristic if not present
        best_model_path = os.path.join(results_dir, 'task3_best_tactic_per_product_model.csv')
        best_heur_path = os.path.join(results_dir, 'task3_best_tactic_per_product.csv')
        full_model_path = os.path.join(results_dir, 'task3_tactic_scores_model.csv')
        full_heur_path = os.path.join(results_dir, 'task3_tactic_scores_full.csv')

        if os.path.exists(best_model_path):
            best_df = pd.read_csv(best_model_path)
        elif os.path.exists(best_heur_path):
            best_df = pd.read_csv(best_heur_path)
        else:
            best_df = None

        if os.path.exists(full_model_path):
            full_df = pd.read_csv(full_model_path)
        elif os.path.exists(full_heur_path):
            full_df = pd.read_csv(full_heur_path)
        else:
            full_df = None

        if best_df is None or full_df is None or best_df.empty or full_df.empty:
            st.warning("Task 3 outputs not found. Run scripts/training/task3_best_tactic_classifier.py")
        else:
            products = sorted(best_df['PRODUCT_GROUP'].unique())
            prod_options = [p for p in products if (prod_search.strip().lower() in p.lower())] if prod_search else products
            sel_product = st.selectbox("Product Group", prod_options)

            # Recommended tactic (top-1)
            rec = best_df[best_df['PRODUCT_GROUP'] == sel_product]
            st.write("Recommended tactic for selected product:")
            display_cols = [c for c in [
                'PRODUCT_GROUP','TACTIC_TYPE','pred_approval_prob','pred_expected_score',
                'score','approval_rate','claim_count','total_claim_amount'
            ] if c in rec.columns]
            st.dataframe(rec[display_cols], hide_index=True, width='stretch')
            if not rec.empty:
                mm1, mm2 = st.columns(2)
                if 'pred_approval_prob' in rec.columns:
                    mm1.metric("Pred. Approval", f"{float(rec['pred_approval_prob'].iloc[0]):.2f}")
                rank_metric = 'pred_expected_score' if 'pred_expected_score' in rec.columns else ('score' if 'score' in rec.columns else None)
                if rank_metric:
                    mm2.metric("Expected Score", f"{float(rec[rank_metric].iloc[0]):.2f}")

            # Top tactics list for the product
            prod_df = full_df[full_df['PRODUCT_GROUP'] == sel_product].copy()
            rank_col = 'pred_expected_score' if 'pred_expected_score' in prod_df.columns else 'score'
            prod_df = prod_df.sort_values(rank_col, ascending=False)
            show_cols = [c for c in [
                'TACTIC_TYPE','pred_approval_prob','pred_expected_score','score','approval_rate',
                'claim_count','total_claim_amount','approved_count','pending_count','rejected_count'
            ] if c in prod_df.columns]
            st.write("Top tactics by expected effectiveness:")
            st.dataframe(prod_df[show_cols].head(10), hide_index=True, width='stretch')

            # Downloads
            st.download_button(
                label="📥 Download best-per-product",
                data=best_df.to_csv(index=False),
                file_name=os.path.basename(best_model_path if os.path.exists(best_model_path) else best_heur_path),
                mime="text/csv"
            )
            st.download_button(
                label="📥 Download full tactic scores",
                data=full_df.to_csv(index=False),
                file_name=os.path.basename(full_model_path if os.path.exists(full_model_path) else full_heur_path),
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error loading Task 3 outputs: {e}")
        st.code(traceback.format_exc())