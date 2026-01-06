import joblib
import pandas as pd

history = joblib.load('models/history_monthly.pkl')
cust_data = history[(history['CUSTOMER'] == 'AL KABAYEL DISCOUNT CENTRE(BR)-D') & 
                    (history['PRODUCT_GROUP'] == 'Snacks')]

print("Last 24 months of sales:")
print(cust_data[['MONTH_START', 'SALES']].tail(24).to_string())
print(f"\nRecent 12-month avg: ${cust_data.tail(12)['SALES'].mean():.2f}")
print(f"Historical avg: ${cust_data['SALES'].mean():.2f}")
print(f"Max sale: ${cust_data['SALES'].max():.2f}")
print(f"Min sale: ${cust_data['SALES'].min():.2f}")
print(f"Total records: {len(cust_data)}")
