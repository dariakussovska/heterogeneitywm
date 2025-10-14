import os
import glob

def convert_excel_to_feather(directory):
    """Convert all Excel files in directory to Feather format"""
    excel_files = glob.glob(os.path.join(directory, '*.xlsx'))
    
    for excel_file in excel_files:
        # Read Excel
        df = pd.read_excel(excel_file)
        
        # Save as Feather
        feather_file = excel_file.replace('.xlsx', '.feather')
        df.to_feather(feather_file)
        print(f"Converted: {os.path.basename(excel_file)}")
    
    print("Conversion complete! Use .feather files for faster loading.")

# Convert all your files
convert_excel_to_feather('/home/daria/PROJECT/')
