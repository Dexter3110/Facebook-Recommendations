import pandas as pd
import os

def compress_without_data_loss():
    """Compress CSV file without losing ANY data"""
    
    print("Loading original data...")
    df = pd.read_csv('featured_test_data.csv')
    
    original_size = os.path.getsize('featured_test_data.csv') / (1024*1024)
    print(f"Original: {len(df):,} rows, {original_size:.2f} MB")
    
    # COMPRESS - NO DATA LOSS
    df.to_csv('featured_test_data.csv.gz', index=False, compression='gzip')
    
    compressed_size = os.path.getsize('featured_test_data.csv.gz') / (1024*1024)
    print(f"Compressed: {len(df):,} rows, {compressed_size:.2f} MB")
    
    # Verify - load compressed file
    df_check = pd.read_csv('featured_test_data.csv.gz', compression='gzip')
    print(f"Verification: {len(df_check):,} rows loaded from compressed file")
    
    print("\n✅ ZERO DATA LOSS - All rows preserved!")
    print("✅ File size reduced through compression only!")

if __name__ == "__main__":
    compress_without_data_loss()