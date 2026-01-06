import pandas as pd
import numpy as np

monthly_demand = pd.read_csv("D:\\BlueRose\\iffco_discount_model\\outputs\\task2_demand_forecast.csv")

monthly_demand.head()

baseline = (
    monthly_demand
    .groupby(['PRODUCT_GROUP', 'CUSTOMER_CHANNEL'], as_index=False)
    .agg(
        AVG_MONTHLY_CLAIM=('MONTHLY_CLAIM_AMOUNT', 'mean')
    )
)

df = monthly_demand.merge(
    baseline,
    on=['PRODUCT_GROUP', 'CUSTOMER_CHANNEL'],
    how='left'
)

df['PROMO_INTENSITY_INDEX'] = (
    df['MONTHLY_CLAIM_AMOUNT'] / df['AVG_MONTHLY_CLAIM']
)

def classify_promo_window(x):
    if x >= 1.25:
        return 'High Impact'
    elif x >= 0.85:
        return 'Medium Impact'
    else:
        return 'Low Impact'

df['PROMO_WINDOW'] = df['PROMO_INTENSITY_INDEX'].apply(classify_promo_window)

def promo_recommendation(row):
    if row['PROMO_WINDOW'] == 'High Impact':
        return 'Increase promotions'
    elif row['PROMO_WINDOW'] == 'Medium Impact':
        return 'Maintain promotions'
    else:
        return 'Reduce promotions'

df['PROMO_RECOMMENDATION'] = df.apply(promo_recommendation, axis=1)

final_timing = df[
    [
        'YEAR',
        'MONTH_NUM',
        'SEASON',
        'PRODUCT_GROUP',
        'CUSTOMER_CHANNEL',
        'MONTHLY_CLAIM_AMOUNT',
        'PROMO_INTENSITY_INDEX',
        'PROMO_WINDOW',
        'PROMO_RECOMMENDATION'
    ]
].copy()

final_timing.head()

final_timing.to_csv(
    "D:\\BlueRose\\iffco_discount_model\\outputs\\task3_promotional_timing_recommendations.csv",
    index=False
)
