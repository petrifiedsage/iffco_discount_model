import pandas as pd
import os

def split_excel_to_csv(file_path, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Load the Excel file
    # None as sheet_name loads all sheets as a dictionary
    excel_file = pd.read_excel(file_path, sheet_name=None)

    for sheet_name, data in excel_file.items():
        # Clean sheet name to prevent filename errors
        clean_name = "".join([c for c in sheet_name if c.isalnum() or c in (' ', '_')]).strip()
        file_name = f"{clean_name}.csv"
        save_path = os.path.join(output_folder, file_name)
        
        # Save to CSV
        data.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

# Usage
split_excel_to_csv('D:\\BlueRose\\iffco_discount_model\\data\\TDA_FY 22_23.xlsx', 'D:\\BlueRose\\iffco_discount_model\\data')