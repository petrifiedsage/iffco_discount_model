import pandas as pd
import os
import glob

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")

def extract_columns():
    print("--- 🔍 STEP 1: COLUMN & HEADER DISCOVERY ---")
    
    # Search for all Contract/Export files
    files = glob.glob(os.path.join(DATA_DIR, "*Contract*.xls*"))
    
    if not files:
        print("❌ No Contract files found in data folder.")
        return

    for f in files:
        filename = os.path.basename(f)
        print(f"\n📂 FILE: {filename}")
        try:
            # Read first few rows
            if f.lower().endswith('.xls'):
                df = pd.read_excel(f, engine='xlrd', nrows=5)
            else:
                df = pd.read_excel(f, nrows=5)
            
            # Print Columns found at Row 0
            print(f"   🔹 Headers at Row 0: {list(df.columns)}")
            
            # Check if real headers are deeper (Row 1 or 2)
            # Sometimes exports have a title row like "Report generated on..."
            print(f"   🔹 Row 1 Data: {df.iloc[0].tolist()}")
            print(f"   🔹 Row 2 Data: {df.iloc[1].tolist()}")
            
        except Exception as e:
            print(f"   ❌ Error reading: {e}")

if __name__ == "__main__":
    extract_columns()