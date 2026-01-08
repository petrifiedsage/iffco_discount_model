"""
TASK 2: Sales Forecasting Inference Script
Uses corrected model trained on Contract_Claim_Dump with PRODUCT_GROUP (not SKU)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "task2_reboot_model.pkl"
METADATA_PATH = MODEL_DIR / "task2_reboot_metadata.pkl"
SEASONALITY_PATH = MODEL_DIR / "product_seasonality_correct.csv"
STATES_PATH = MODEL_DIR / "task2_reboot_states.pkl"

print("Loading model artifacts...")
try:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    df_seasonality = pd.read_csv(SEASONALITY_PATH)
    states = joblib.load(STATES_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

print(f"  Products: {len(metadata['products'])}")
print(f"  Customers: {len(metadata['customers'])}")
print(f"  Features: {len(metadata['feature_cols'])}")


def forecast_sales(customer_name, product_group, months_ahead=3):
    """
    Forecast sales for a customer-product combination
    
    Parameters:
    -----------
    customer_name : str
        Customer name (e.g., "AL RAYES TRADERS")
    product_group : str
        Product group (e.g., "Ice creams", "Oils", "Beverages")
    months_ahead : int
        Number of months to forecast (1-12)
    
    Returns:
    --------
    dict with forecast data
    """
    
    # Get state for this customer-product pair
    key = (customer_name, product_group)
    
    if key not in states:
        return {
            'status': 'error',
            'message': f'No history for {customer_name} + {product_group}',
            'forecast': []
        }
    
    state = states[key]
    recent_history = np.array(state['recent_history'], dtype=float)
    
    if len(recent_history) == 0:
        return {
            'status': 'error',
            'message': 'No sales history available',
            'forecast': []
        }
    
    # Get seasonality factors
    product_seasonality = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product_group]
    
    forecasts = []
    current_lag_1 = recent_history[-1] if len(recent_history) > 0 else recent_history.mean()
    current_lag_3 = recent_history[-3:].mean() if len(recent_history) >= 3 else recent_history.mean()
    current_lag_6 = recent_history[-6:].mean() if len(recent_history) >= 6 else recent_history.mean()
    current_lag_12 = recent_history[-12:].mean() if len(recent_history) >= 12 else recent_history.mean()
    
    # Use moving average instead of just last value to smooth volatility
    baseline = recent_history[-6:].mean() if len(recent_history) >= 6 else recent_history.mean()
    
    # Get current month
    now = datetime.now()
    current_month = now.month
    
    for step in range(1, months_ahead + 1):
        forecast_month = ((current_month + step - 1) % 12) + 1
        
        # Get seasonality for this month
        season_data = product_seasonality[product_seasonality['MONTH'] == forecast_month]
        seasonality_index = season_data['SEASONALITY_INDEX'].values[0] if len(season_data) > 0 else 1.0
        
        # Create features for prediction
        quarter = ((forecast_month - 1) // 3) + 1
        is_summer = 1 if forecast_month in [6, 7, 8] else 0
        year = now.year if step <= (12 - current_month + 1) else now.year + 1
        
        # Trend calculation (momentum)
        trend_3m = (current_lag_1 - current_lag_3) / 3 if current_lag_3 > 0 else 0
        
        # Build feature vector
        features = {
            'LAG_1': current_lag_1,
            'LAG_3': current_lag_3,
            'LAG_6': current_lag_6,
            'LAG_12': current_lag_12,
            'LAG_3_AVG': current_lag_3,
            'LAG_6_AVG': current_lag_6,
            'CUSTOMER_AVG': state['customer_avg'],
            'PRODUCT_AVG': state.get('product_avg', state['customer_avg']),
            'CUSTOMER_PRODUCT_AVG': state['customer_avg'],
            'TREND_3M': trend_3m,
            'MONTH': forecast_month,
            'QUARTER': quarter,
            'YEAR': year,
            'IS_SUMMER': is_summer,
            'SEASONALITY_INDEX': seasonality_index,
            # Promo features (set to 0 for forecast period - assuming no promos)
            'PROMO_AMOUNT': 0,
            'PROMO_COUNT': 0,
            'PROMO_LOG': 0,
            'PROMO_COUNT_LOG': 0,
            'PROMO_3M_AVG': 0
        }
        
        # Ensure feature order matches training
        X = pd.DataFrame([features])[metadata['feature_cols']]
        
        # Predict (output is log-transformed)
        y_pred_log = model.predict(X)[0]
        y_pred_actual = np.expm1(y_pred_log)
        
        # Apply seasonality with confidence weighting
        season_confidence = season_data['CONFIDENCE'].values[0] if len(season_data) > 0 else 0.5
        if season_confidence > 0.7:
            # High confidence: apply full seasonality, but blend with baseline to avoid extreme drops
            final_forecast = (y_pred_actual * seasonality_index * 0.7) + (baseline * seasonality_index * 0.3)
        else:
            # Low confidence: more conservative seasonality
            final_forecast = y_pred_actual * (0.5 + 0.5 * seasonality_index)
        
        forecasts.append({
            'month': forecast_month,
            'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][forecast_month - 1],
            'forecast': round(max(0, final_forecast), 2),
            'seasonality': round(seasonality_index, 3),
            'confidence': round(season_confidence, 2)
        })
        
        # Update lags for next iteration (recursive)
        current_lag_12 = current_lag_6
        current_lag_6 = current_lag_3
        current_lag_3 = (current_lag_3 * 2 + final_forecast) / 3  # Exponential moving average
        current_lag_1 = final_forecast
    
    return {
        'status': 'success',
        'customer': customer_name,
        'product': product_group,
        'forecast': forecasts
    }


if __name__ == "__main__":
    # Example: Test with a sample customer-product combination
    print("\n" + "="*70)
    print("EXAMPLE FORECAST")
    print("="*70)
    
    # Get first available customer-product from states
    sample_key = list(states.keys())[0]
    customer, product = sample_key
    
    print(f"\nCustomer: {customer}")
    print(f"Product: {product}")
    print(f"\nForecasting next 6 months:\n")
    
    result = forecast_sales(customer, product, months_ahead=6)
    
    if result['status'] == 'success':
        for fc in result['forecast']:
            print(f"  {fc['month_name']:>3} (Month {fc['month']:2d}): ${fc['forecast']:>10,.2f}  "
                  f"(seasonality: {fc['seasonality']:.2f}x, confidence: {fc['confidence']:.1%})")
    else:
        print(f"  {result['message']}")
    
    print("\n" + "="*70)
    print("Model ready for Streamlit deployment!")
    print("="*70)
