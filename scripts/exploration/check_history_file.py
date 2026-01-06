import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
MODELS_DIR = os.path.join(BASE_DIR, "models")
HIST_PATH = os.path.join(MODELS_DIR, 'history_monthly.pkl')

def check_history_content():
    print(f"--- Diagnosing History File: {HIST_PATH} ---")
    
    if not os.path.exists(HIST_PATH):
        print("❌ Error: File does not exist!")
        return

    try:
        df = joblib.load(HIST_PATH)
        print(f"✅ File Loaded Successfully. Total Rows: {len(df):,}")
        
        # 1. Check Unique Customers
        unique_custs = df['CUSTOMER'].nunique()
        print(f"   📊 Unique Customers Found: {unique_custs}")
        
        # 2. Show Top 5 Customers by Name
        print("\n   🔎 First 5 Customers in File:")
        print(df['CUSTOMER'].unique()[:5])
        
        # 3. Check for specific bad pattern
        bad_cust = "1 To 10 Center General Trading L.Lc-F"
        count_bad = len(df[df['CUSTOMER'] == bad_cust])
        print(f"\n   ℹ️ Rows for '{bad_cust}': {count_bad}")
        
        if unique_custs < 10:
            print("\n❌ CRITICAL: The file contains very few customers. The 'create_history_file.py' script failed to load all data.")
        else:
            print("\n✅ The file looks healthy. The issue is likely in the App's filtering logic.")

    except Exception as e:
        print(f"❌ Error Reading File: {e}")

if __name__ == "__main__":
    check_history_content()