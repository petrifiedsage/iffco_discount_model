import pandas as pd
from pathlib import Path

# 1. Define the input path
input_path = Path(r'D:\BlueRose\iffco_discount_model\data\Contract_Claim_Dump-Jan22-Dec22-UAE.xls')

# 2. Define the output path in the F1000 subfolder
output_folder = input_path.parent / "F1000"
output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist
output_path = output_folder / f"{input_path.stem}_F1000.xlsx"

# 3. Load the Excel structure
excel_data = pd.ExcelFile(input_path)

print(f"Reading from: {input_path.name}")
print(f"Creating new file: {output_path.name}...")

# 4. Use ExcelWriter to save multiple sheets into a single workbook
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet_name in excel_data.sheet_names:
        # Read only the first 1000 rows
        df = pd.read_excel(input_path, sheet_name=sheet_name, nrows=1000)
        
        # Define the new sheet name with the tag
        new_sheet_name = f"{sheet_name}_F1000"
        
        # Excel has a 31-character limit for sheet names
        # This slice ensures the name isn't too long, which would cause an error
        final_sheet_name = new_sheet_name[:31]
        
        # Write to the same workbook
        df.to_excel(writer, sheet_name=final_sheet_name, index=False)
        print(f"  - Added sheet: {final_sheet_name}")

print(f"\nDone! Your file is ready at:\n{output_path}")