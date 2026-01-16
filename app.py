import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import datetime
import traceback
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="IFFCO Trade Strategy Engine",
    page_icon="🧊",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

st.title("🚀 IFFCO Intelligent Trade Planner")

# --- SIDEBAR ORGANIZATION ---
st.sidebar.write("### 🎯 Navigation")
st.sidebar.write("---")
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
    st.subheader("💰 Optimal Discount Recommendation")
    st.caption("AI-powered discount optimization with guardrails")
    
    if disc_meta:
        with st.expander("📋 Input Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**👥 Customer & Account**")
                search_customer = st.text_input("🔍 Search Customer", placeholder="Type to filter...", key="search_cust")
                filtered_customers = [c for c in disc_meta['customers'] if search_customer.lower() in c.lower()] if search_customer else disc_meta['customers']
                customer = st.selectbox("Select Customer", filtered_customers, key="sel_cust")
                global_acc = disc_meta['cust_global_map'].get(customer, "Unknown")
                st.caption(f"📌 Account: {global_acc}")
                channel = st.selectbox("Channel", disc_meta['channels'], key="channel")
            
            with col2:
                st.write("**📦 Product & Tactic**")
                search_product = st.text_input("🔍 Search Product", placeholder="Type to filter...", key="search_prod")
                filtered_products = [p for p in disc_meta['products'] if search_product.lower() in p.lower()] if search_product else disc_meta['products']
                product = st.selectbox("Product Group", filtered_products, key="product")
                
                search_tactic = st.text_input("🔍 Search Tactic", placeholder="Type to filter...", key="search_tact")
                filtered_tactics = [t for t in disc_meta['tactics'] if search_tactic.lower() in t.lower()] if search_tactic else disc_meta['tactics']
                tactic = st.selectbox("Tactic Type", filtered_tactics, key="tactic")
                
                ctype = st.selectbox("Contract Type", disc_meta['types'], key="ctype")
            
            with col3:
                st.write("**📅 Timeline & Duration**")
                start_date = st.date_input("Contract Start", datetime.date(2026, 1, 1), key="start_date")
                duration = st.slider("Duration (Months)", 1, 12, 6, key="duration")
                end_month = (start_date.month + duration) % 12 or 12
                st.caption(f"📌 Ends: Month {end_month}")
        
        # Full-width calculate button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            calculate_btn = st.button("🤖 Calculate Optimal Discount", type="primary", use_container_width=True)
        
        if calculate_btn:
            input_data = {
                'CUSTOMER': [customer], 'GLOBAL_ACCOUNT': [global_acc],
                'CHANNEL': [channel], 'TYPE': [ctype],
                'TACTIC_TYPE': [tactic], 'PRODUCT_GROUP': [product],
                'START_MONTH_NUM': [start_date.month],
                'END_MONTH_NUM': [end_month]
            }
            try:
                with st.spinner("🔄 Computing optimal discount..."):
                    time.sleep(0.3)
                    prediction = discount_model.predict(pd.DataFrame(input_data))[0]
                
                st.divider()
                
                # Result card with color-coded background
                if prediction > 14.0:
                    color = "#FFE5E5"
                    emoji = "⚠️"
                    message = "High Discount Alert"
                    advice = "Consider renegotiating terms. Margin impact: HIGH - Verify margin protection policies."
                    status = "caution"
                elif prediction < 11.0:
                    color = "#E5FFE5"
                    emoji = "✅"
                    message = "Strong Profitability"
                    advice = "Discount is safe. Good margin preserved. Competitive positioning maintained."
                    status = "success"
                else:
                    color = "#E5F5FF"
                    emoji = "ℹ️"
                    message = "Standard Range"
                    advice = "Competitive discount within normal limits. Balanced profitability and market positioning."
                    status = "info"
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 25px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h3>{emoji} {message}</h3>
                <h2 style="margin: 15px 0; color: #1f77b4;">{prediction:.2f}%</h2>
                <p style="margin: 10px 0; font-size: 16px;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Additional metrics
                st.subheader("📊 Discount Benchmark")
                bench_col1, bench_col2, bench_col3 = st.columns(3)
                with bench_col1:
                    st.metric("Your Discount", f"{prediction:.2f}%")
                with bench_col2:
                    st.metric("Safety Range", "10-15%", f"{'+' if prediction < 15 else '-'} {abs(prediction - 12.5):.1f}%")
                with bench_col3:
                    st.metric("Risk Level", "Low" if prediction <= 14 else "High", "Approved" if prediction <= 14 else "Review")
                
                # Export button
                col_exp1, col_exp2 = st.columns([3, 1])
                with col_exp2:
                    export_data = f"Customer: {customer}\nProduct: {product}\nTactic: {tactic}\nRecommended Discount: {prediction:.2f}%"
                    st.download_button(
                        "📥 Export Result",
                        data=export_data,
                        file_name="discount_recommendation.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
            except ValueError as e:
                st.error("❌ Input Error: Some fields contain invalid values")
                st.info("💡 Tip: Ensure all required fields are filled with valid data from the dropdowns")
            except Exception as e:
                st.error(f"❌ Calculation Error")
                st.warning("📞 Please verify your inputs and try again")
                with st.expander("🔧 Technical Details"):
                    st.code(str(e))
    else:
        st.error("❌ Model configuration not loaded. Please ensure models are in the /models directory.")
# ==========================================
# TAB 2: SALES FORECASTING (CORRECTED MODEL)
# ==========================================
with tab2:
    st.subheader("📈 Sales Forecasting")
    st.caption("Uses Contract_Claim_Dump data with actual product categories")

    tab_forecast, tab_val2025, tab_forecast2026 = st.tabs(["Customer Based Forecast", "2025 Validation", "2026 Forecast"])

    with tab_forecast:
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
                        st.info(f"💡 Tip: Try other customer-product combinations. Available: {len(states)} pairs")
                    else:
                        with st.spinner("🔄 Generating forecast..."):
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
                            m1.metric("Total Forecast", f"${total_forecast:,.0f}")
                            m2.metric("Avg Monthly", f"${total_forecast/len(forecasts):,.0f}")
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
                            fig.update_layout(title=f"Sales Forecast: {selected_product}", xaxis_title="Date", yaxis_title="Sales (AED)", hovermode='x unified', height=400)
                            st.plotly_chart(fig, use_container_width=True)

                            st.dataframe(
                                forecast_df[['month_name', 'forecast', 'seasonality', 'confidence']]
                                .rename(columns={'month_name': 'Month', 'forecast': 'Forecast (AED)', 'seasonality': 'Seasonality', 'confidence': 'Confidence'})
                                .style.format({'Forecast (AED)': '${:,.2f}', 'Seasonality': '{:.2f}x', 'Confidence': '{:.0%}'}),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.error(result['message'])
                except ValueError as e:
                    st.error("❌ Input Error: Invalid customer or product selected")
                    st.info("💡 Tip: Ensure the customer-product combination has historical data")
                except Exception as e:
                    st.error(f"❌ Forecast Generation Error")
                    st.warning("📞 Please try another customer-product combination")
                    with st.expander("🔧 Technical Details"):
                        st.code(traceback.format_exc())
        else:
            st.error("❌ Model not loaded. Please run train_task2_corrected.py first.")

    with tab_val2025:
        st.subheader("📊 2025 Validation Results")
        st.caption("Model validation: Predicted vs Actual 2025 data (Retrained on 2022-2024)")

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
                st.metric("Actual Total (2025)", f"${val_df['TOTAL_SALES'].sum():,.0f}")
            with m4:
                st.metric("Predicted Total (2025)", f"${val_df['PREDICTED_SALES'].sum():,.0f}")

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
            display_val['TOTAL_SALES'] = display_val['TOTAL_SALES'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
            display_val['PREDICTED_SALES'] = display_val['PREDICTED_SALES'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
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
        st.subheader("📊 2026 Sales Forecast")
        st.caption("Forecast based on 2022-2024 training data, validated on 2025")

        if forecast_2026 is not None and len(forecast_2026) > 0:
            forecast_sorted = forecast_2026.copy()

            numeric_cols = ['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']
            for col in numeric_cols:
                if col in forecast_sorted.columns:
                    forecast_sorted[col] = pd.to_numeric(forecast_sorted[col], errors='coerce')

            if 'TOTAL_2026' in forecast_sorted.columns:
                forecast_sorted = forecast_sorted.sort_values('TOTAL_2026', ascending=False)

            m1, m2 = st.columns(2)
            with m1:
                total_2026 = forecast_2026[numeric_cols].sum().sum() if numeric_cols[0] in forecast_2026.columns else 0
                st.metric("Total 2026 Forecast", f"${total_2026:,.0f}")
            with m2:
                st.metric("Products Forecasted", f"{len(forecast_2026)}")
    

            st.write("**Forecast by Product and Season (All Products):**")
            display_full = forecast_sorted[['PRODUCT_GROUP'] + [c for c in numeric_cols if c in forecast_sorted.columns]].copy()
            for col in numeric_cols:
                if col in display_full.columns:
                    display_full[col] = display_full[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

            st.dataframe(display_full, hide_index=True, width='stretch')

            st.download_button(label="📥 Download 2026 Forecast", data=forecast_2026.to_csv(index=False), file_name="task2_2026_forecast_retrained.csv", mime="text/csv")
        else:
            st.info("Run scripts/task2_retrain_2024_validate_2025.py to generate 2026 forecast.")
    
# ==========================================
# TAB 3: STRATEGY MATRIX
# ==========================================
with tab3:
    st.subheader("🎯 Best Tactic Recommendation per Product")
    st.caption("Data-driven recommendations based on historical effectiveness")
    try:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        # Prefer new 2026 model outputs, fallback to legacy model, then heuristic
        new_best_path = os.path.join(results_dir, 'task3_2026_tactic_recommendations.csv')
        new_full_path = os.path.join(results_dir, 'task3_2026_tactic_scores.csv')
        best_model_path = os.path.join(results_dir, 'task3_best_tactic_per_product_model.csv')
        best_heur_path = os.path.join(results_dir, 'task3_best_tactic_per_product.csv')
        full_model_path = os.path.join(results_dir, 'task3_tactic_scores_model.csv')
        full_heur_path = os.path.join(results_dir, 'task3_tactic_scores_full.csv')

        if os.path.exists(new_best_path):
            best_df = pd.read_csv(new_best_path)
        elif os.path.exists(best_model_path):
            best_df = pd.read_csv(best_model_path)
        elif os.path.exists(best_heur_path):
            best_df = pd.read_csv(best_heur_path)
        else:
            best_df = None

        if os.path.exists(new_full_path):
            full_df = pd.read_csv(new_full_path)
        elif os.path.exists(full_model_path):
            full_df = pd.read_csv(full_model_path)
        elif os.path.exists(full_heur_path):
            full_df = pd.read_csv(full_heur_path)
        else:
            full_df = None

        if best_df is None or full_df is None or best_df.empty or full_df.empty:
            st.warning("Task 3 outputs not found. Run scripts/training/task3_best_tactic_classifier.py")
        else:
            products = sorted(best_df['PRODUCT_GROUP'].unique())
            prod_search = st.text_input("🔍 Search Product Group", placeholder="Type to filter...")
            prod_options = [p for p in products if (prod_search.strip().lower() in p.lower())] if prod_search else products
            sel_product = st.selectbox("Product Group", prod_options)

            # Recommended tactic (top-1)
            rec = best_df[best_df['PRODUCT_GROUP'] == sel_product]
            
            if not rec.empty:
                st.subheader("⭐ Recommended Tactic")
                
                # Create a nice card for the recommendation
                rec_row = rec.iloc[0]
                tactic_name = rec_row['TACTIC_TYPE']
                # Prefer 2026 predicted expected score, fallback to legacy
                if 'pred_expected_score_2026' in rec.columns and pd.notnull(rec_row.get('pred_expected_score_2026', None)):
                    eff_score = rec_row.get('pred_expected_score_2026')
                elif 'pred_expected_score' in rec.columns and pd.notnull(rec_row.get('pred_expected_score', None)):
                    eff_score = rec_row.get('pred_expected_score')
                else:
                    eff_score = rec_row.get('score', 0)
                approval = rec_row.get('approval_rate', rec_row.get('pred_approval_prob', 0))
                
                # Get detailed metrics from full_df (has approved/pending breakdown)
                detail_row = full_df[(full_df['PRODUCT_GROUP'] == sel_product) & (full_df['TACTIC_TYPE'] == tactic_name)]
                if not detail_row.empty:
                    detail_row = detail_row.iloc[0]
                else:
                    detail_row = rec_row
                
                col_card1, col_card2, col_card3 = st.columns([2, 1, 1])
                with col_card1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                    <h3 style="margin: 0; color: white;">🏆 {tactic_name}</h3>
                    <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">Best performing tactic for {sel_product}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_card2:
                    st.metric("Effectiveness", f"{eff_score:.1f}", help="Composite effectiveness score (higher is better)")
                
                with col_card3:
                    st.metric("Approval Rate", f"{approval:.1%}", help="Successful rate")
                
                st.divider()
                
                # Detailed metrics
                st.subheader("📊 Detailed Metrics")
                
                # Show approval breakdown with explanation
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Claims Count", f"{int(detail_row.get('claim_count', 0)):,}")
                with metric_col2:
                    st.metric("Total Amount", f"${detail_row.get('total_claim_amount', 0):,.0f}")
                with metric_col3:
                    approved = int(detail_row.get('approved_count', 0))
                    st.metric("Approved", f"{approved:,}", help="Fully approved claims")
                with metric_col4:
                    pending = int(detail_row.get('pending_count', 0))
                    st.metric("Pending", f"{pending:,}", help="Claims awaiting approval")
                
                # Approval rate explanation
                if detail_row.get('pending_count', 0) > 0:
                    st.caption(f"ℹ️ Approval Rate = (Approved + 0.5 × Pending) ÷ Total = ({approved} + 0.5 × {pending}) ÷ {int(detail_row.get('claim_count', 0))} = {approval:.1%}")
                
                st.divider()
                
                st.subheader("📈 All Tactics Ranked")
                
                # Top tactics list for the product
                prod_df = full_df[full_df['PRODUCT_GROUP'] == sel_product].copy()
                # Prefer 2026 rank column when available
                if 'pred_expected_score_2026' in prod_df.columns:
                    rank_col = 'pred_expected_score_2026'
                elif 'pred_expected_score' in prod_df.columns:
                    rank_col = 'pred_expected_score'
                else:
                    rank_col = 'score'
                prod_df = prod_df.sort_values(rank_col, ascending=False)
                show_cols = [c for c in [
                    'TACTIC_TYPE','pred_approval_prob','pred_expected_score','score','approval_rate',
                    'claim_count','total_claim_amount'
                ] if c in prod_df.columns]
                
                top_n = st.slider("Show top N tactics", 3, 20, 10, help="Adjust to see more or fewer tactics")
                
                # Format dataframe for display
                display_df = prod_df[show_cols].head(top_n).copy()
                display_df.index = range(1, len(display_df) + 1)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.divider()
                
                # Downloads
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "📥 Download Best Tactic",
                        data=best_df.to_csv(index=False),
                        file_name="best_tactic_per_product.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    st.download_button(
                        "📥 Download All Tactics",
                        data=full_df.to_csv(index=False),
                        file_name="all_tactics_ranked.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"❌ Error Loading Task 3")
        st.warning("📞 Please ensure Task 3 training has been completed")
        with st.expander("🔧 Technical Details"):
            st.code(traceback.format_exc())