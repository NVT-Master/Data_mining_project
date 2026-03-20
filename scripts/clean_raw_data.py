"""
Data Cleaning Script
Làm sạch dữ liệu gốc từ data/raw/
Tạo file cleaned trong data/raw/cleaned/
File gốc được giữ nguyên
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
RAW_DATA_DIR = Path('../data/raw')
CLEANED_DATA_DIR = RAW_DATA_DIR / 'cleaned'
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🧹 DATA CLEANING - GỮ FILE GỐC, TẠO FILE CLEANED")
print("=" * 80)

# ============================================================================
# 1. CLEAN TEMP.CSV
# ============================================================================
print("\n\n1️⃣  CLEANING: temp.csv")
print("-" * 80)

temp_file = RAW_DATA_DIR / 'temp.csv'
temp_df = pd.read_csv(temp_file)

print(f"   📊 Original shape: {temp_df.shape}")
print(f"   ❌ Missing values BEFORE:")
print(f"      • year: {temp_df['year'].isna().sum()}")
print(f"      • country: {temp_df['country'].isna().sum()}")
print(f"      • avg_temp: {temp_df['avg_temp'].isna().sum()}")

# Clean temp.csv
# Remove rows with missing avg_temp (important column)
temp_df_clean = temp_df.dropna(subset=['avg_temp']).copy()

# Convert avg_temp to numeric (in case of formatting issues)
temp_df_clean['avg_temp'] = pd.to_numeric(temp_df_clean['avg_temp'], errors='coerce')

# Remove rows where conversion failed
temp_df_clean = temp_df_clean.dropna(subset=['avg_temp'])

print(f"\n   ✅ Cleaned shape: {temp_df_clean.shape}")
print(f"   ✅ Missing values AFTER:")
print(f"      • year: {temp_df_clean['year'].isna().sum()}")
print(f"      • country: {temp_df_clean['country'].isna().sum()}")
print(f"      • avg_temp: {temp_df_clean['avg_temp'].isna().sum()}")
print(f"   📊 Rows removed: {temp_df.shape[0] - temp_df_clean.shape[0]}")

# Save cleaned
temp_clean_file = CLEANED_DATA_DIR / 'temp_cleaned.csv'
temp_df_clean.to_csv(temp_clean_file, index=False)
print(f"   ✓ Saved: {temp_clean_file.name}")

# ============================================================================
# 2. CLEAN RAINFALL.CSV
# ============================================================================
print("\n\n2️⃣  CLEANING: rainfall.csv")
print("-" * 80)

rainfall_file = RAW_DATA_DIR / 'rainfall.csv'
rainfall_df = pd.read_csv(rainfall_file)

print(f"   📊 Original shape: {rainfall_df.shape}")
print(f"   ❌ Missing values & issues BEFORE:")
missing_rain = rainfall_df['average_rain_fall_mm_per_year'].isna().sum()
empty_rain = (rainfall_df['average_rain_fall_mm_per_year'] == '').sum()
dots_rain = (rainfall_df['average_rain_fall_mm_per_year'] == '..').sum()

print(f"      • NaN: {missing_rain}")
print(f"      • Empty string: {empty_rain}")
print(f"      • '..' values: {dots_rain}")

# Clean rainfall.csv
# Replace '..' and empty strings with NaN
rainfall_df_clean = rainfall_df.copy()
rainfall_df_clean['average_rain_fall_mm_per_year'] = rainfall_df_clean['average_rain_fall_mm_per_year'].replace(['..', '', ' '], np.nan)

# Convert to numeric
rainfall_df_clean['average_rain_fall_mm_per_year'] = pd.to_numeric(
    rainfall_df_clean['average_rain_fall_mm_per_year'], 
    errors='coerce'
)

# Remove rows with missing rainfall data
rainfall_df_clean = rainfall_df_clean.dropna(subset=['average_rain_fall_mm_per_year']).copy()

print(f"\n   ✅ Cleaned shape: {rainfall_df_clean.shape}")
print(f"   ✅ Missing values AFTER:")
print(f"      • NaN: {rainfall_df_clean['average_rain_fall_mm_per_year'].isna().sum()}")
print(f"   📊 Rows removed: {rainfall_df.shape[0] - rainfall_df_clean.shape[0]}")

# Save cleaned
rainfall_clean_file = CLEANED_DATA_DIR / 'rainfall_cleaned.csv'
rainfall_df_clean.to_csv(rainfall_clean_file, index=False)
print(f"   ✓ Saved: {rainfall_clean_file.name}")

# ============================================================================
# 3. CLEAN PESTICIDES.CSV
# ============================================================================
print("\n\n3️⃣  CLEANING: pesticides.csv")
print("-" * 80)

pesticides_file = RAW_DATA_DIR / 'pesticides.csv'
pesticides_df = pd.read_csv(pesticides_file)

print(f"   📊 Original shape: {pesticides_df.shape}")
print(f"   ❌ Missing values BEFORE:")
print(f"      • Value: {pesticides_df['Value'].isna().sum()}")

# Clean pesticides.csv
pesticides_df_clean = pesticides_df.copy()

# Convert Value to numeric
pesticides_df_clean['Value'] = pd.to_numeric(pesticides_df_clean['Value'], errors='coerce')

# Remove rows with missing values
pesticides_df_clean = pesticides_df_clean.dropna(subset=['Value']).copy()

print(f"\n   ✅ Cleaned shape: {pesticides_df_clean.shape}")
print(f"   ✅ Missing values AFTER:")
print(f"      • Value: {pesticides_df_clean['Value'].isna().sum()}")
print(f"   📊 Rows removed: {pesticides_df.shape[0] - pesticides_df_clean.shape[0]}")

# Save cleaned
pesticides_clean_file = CLEANED_DATA_DIR / 'pesticides_cleaned.csv'
pesticides_df_clean.to_csv(pesticides_clean_file, index=False)
print(f"   ✓ Saved: {pesticides_clean_file.name}")

# ============================================================================
# 4. CLEAN YIELD.CSV (or YIELD_DF.CSV if exists)
# ============================================================================
print("\n\n4️⃣  CLEANING: yield data")
print("-" * 80)

# Try yield_df.csv first, then yield.csv
yield_file = RAW_DATA_DIR / 'yield_df.csv'
if not yield_file.exists():
    yield_file = RAW_DATA_DIR / 'yield.csv'

if yield_file.exists():
    yield_df = pd.read_csv(yield_file)
    
    print(f"   📊 Original shape: {yield_df.shape}")
    print(f"   Columns: {yield_df.columns.tolist()}")
    print(f"   ❌ Missing values BEFORE:")
    for col in yield_df.columns:
        missing = yield_df[col].isna().sum()
        if missing > 0:
            print(f"      • {col}: {missing}")
    
    # Clean yield data
    yield_df_clean = yield_df.copy()
    
    # Find numeric column (likely Yield)
    numeric_cols = yield_df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Remove rows with missing target variable
        yield_df_clean = yield_df_clean.dropna(subset=numeric_cols).copy()
    
    print(f"\n   ✅ Cleaned shape: {yield_df_clean.shape}")
    print(f"   ✅ Missing values AFTER:")
    for col in yield_df_clean.columns:
        missing = yield_df_clean[col].isna().sum()
        if missing > 0:
            print(f"      • {col}: {missing}")
    
    print(f"   📊 Rows removed: {yield_df.shape[0] - yield_df_clean.shape[0]}")
    
    # Save cleaned
    yield_clean_file = CLEANED_DATA_DIR / f'{yield_file.stem}_cleaned.csv'
    yield_df_clean.to_csv(yield_clean_file, index=False)
    print(f"   ✓ Saved: {yield_clean_file.name}")
else:
    print("   ⚠️  No yield file found")

# ============================================================================
# 5. SUMMARY REPORT
# ============================================================================
print("\n\n" + "=" * 80)
print("✅ CLEANING COMPLETED!")
print("=" * 80)

print(f"\n📁 Cleaned files saved in: {CLEANED_DATA_DIR}")
print(f"\nFiles created:")
print(f"   ✓ temp_cleaned.csv ({temp_df_clean.shape[0]} rows)")
print(f"   ✓ rainfall_cleaned.csv ({rainfall_df_clean.shape[0]} rows)")
print(f"   ✓ pesticides_cleaned.csv ({pesticides_df_clean.shape[0]} rows)")
if yield_file.exists():
    print(f"   ✓ {yield_file.stem}_cleaned.csv ({yield_df_clean.shape[0]} rows)")

print(f"\n🔒 Original files in {RAW_DATA_DIR} are UNCHANGED")

print("\n\n🚀 Next steps:")
print("   1. Update loader.py to use cleaned files (or keep using original)")
print("   2. Run pipeline: python scripts/run_pipeline.py")
print("\n" + "=" * 80)
