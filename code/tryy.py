from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure data folder exists
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"📁 Created '{data_folder}' folder")

def clean_price(price_text):
    """Clean price text by removing $ sign and any extra characters"""
    if not price_text or price_text == "Not found":
        return ""
    
    # Remove dollar sign and any leading/trailing whitespace
    cleaned = price_text.replace('$', '').strip()
    
    # Handle price ranges (e.g., "$199.99 to $299.99")
    if 'to' in cleaned.lower():
        parts = cleaned.lower().split('to')
        cleaned = parts[0].strip()
    
    # Remove any non-numeric characters except decimal point
    # This keeps only numbers and decimal points
    import re
    # Find all numbers including decimals
    numbers = re.findall(r'\d+\.?\d*', cleaned)
    if numbers:
        return numbers[0]
    
    return cleaned

def get_ebay_url(product_name):
    """Generate eBay search URL for the given product name"""
    # Clean the product name for URL
    cleaned_name = product_name.strip().replace(' ', '+')
    url = f"https://www.ebay.com/sch/i.html?_nkw={cleaned_name}&_sacat=0&_from=R40&_trksid=p4624852.m570.l1313"
    return url

def extract_features_from_title(title):
    """
    Extract features from product title using regex patterns
    Returns: model_name, storage_gb, model_year, has_defect, condition_score
    """
    # Initialize with default values
    model_name = "Unknown"
    storage_gb = 64  # Default
    model_year = 2020  # Default
    has_defect = 0
    condition_score = 3  # Default "Good" condition
    
    title_lower = title.lower()
    
    # Extract iPhone model
    if 'iphone' in title_lower:
        model_patterns = [
            ('iphone 15', 2023), ('iphone 14', 2022), ('iphone 13', 2021),
            ('iphone 12', 2020), ('iphone 11', 2019), ('iphone x', 2017),
            ('iphone 8', 2017), ('iphone 7', 2016), ('iphone 6', 2014)
        ]
        
        for pattern, year in model_patterns:
            if pattern in title_lower:
                model_name = pattern.title()
                model_year = year
                break
    
    # Extract storage capacity
    storage_pattern = r'(\d+)\s*(gb|tb|mb)'
    storage_match = re.search(storage_pattern, title_lower)
    if storage_match:
        storage_value = int(storage_match.group(1))
        unit = storage_match.group(2)
        
        if unit == 'tb':
            storage_gb = storage_value * 1024
        elif unit == 'gb':
            storage_gb = storage_value
        elif unit == 'mb':
            storage_gb = storage_value / 1024
    
    # Check for defects
    defect_keywords = ['cracked', 'broken', 'damaged', 'not working', 'for parts', 'bad', 'faulty']
    has_defect = any(keyword in title_lower for keyword in defect_keywords)
    
    return model_name, storage_gb, model_year, has_defect, condition_score

def quantify_condition(condition_text):
    """
    Convert subjective condition text to numerical score (1-5)
    5: New/Sealed, 4: Like New, 3: Good, 2: Fair, 1: Poor/For Parts
    """
    condition_lower = condition_text.lower()
    
    if any(word in condition_lower for word in ['new', 'sealed', 'brand new', 'mint']):
        return 5
    elif any(word in condition_lower for word in ['like new', 'excellent', 'perfect']):
        return 4
    elif any(word in condition_lower for word in ['good', 'very good', 'great']):
        return 3
    elif any(word in condition_lower for word in ['fair', 'acceptable', 'ok']):
        return 2
    elif any(word in condition_lower for word in ['poor', 'parts', 'broken', 'damaged']):
        return 1
    else:
        return 3  # Default to "Good"

def remove_outliers_iqr(df, column):
    """
    Remove outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only the non-outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    print(f"📊 Outlier Removal ({column}):")
    print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Removed {len(df) - len(df_clean)} outliers")
    
    return df_clean

def create_joint_collection(products, product_name):
    """
    Create joint_data_collection.csv with cleaned and engineered features
    """
    print("\n" + "="*80)
    print("🛠️  CREATING JOINT DATA COLLECTION")
    print("="*80)
    
    if not products:
        print("❌ No products to process")
        return None
    
    # Create initial DataFrame
    df = pd.DataFrame(products)[['title', 'price', 'condition']]
    
    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Drop rows with missing prices
    initial_count = len(df)
    df = df.dropna(subset=['price'])
    print(f"📈 Initial products: {initial_count}")
    print(f"📉 After dropping NaN prices: {len(df)}")
    
    # Feature Engineering
    print("\n🔧 Feature Engineering:")
    
    # Extract features from title
    features = df['title'].apply(extract_features_from_title)
    df[['model_name', 'storage_gb', 'model_year', 'has_defect', 'condition_score']] = pd.DataFrame(
        features.tolist(), index=df.index
    )
    
    # Quantify condition from condition text
    df['condition_score_from_text'] = df['condition'].apply(quantify_condition)
    
    # Calculate device age
    current_year = 2024
    df['device_age'] = current_year - df['model_year']
    
    # Add shipping cost (random for now)
    df['shipping_cost'] = np.random.choice([0, 5, 10, 15], len(df))
    
    # Combine condition scores (average of both)
    df['final_condition_score'] = (df['condition_score'] + df['condition_score_from_text']) / 2
    
    # Drop unnecessary columns
    df = df.drop(['condition_score', 'condition_score_from_text'], axis=1)
    
    # Remove outliers from price
    df_clean = remove_outliers_iqr(df, 'price')
    
    # Prepare final DataFrame
    # Reorder columns for better readability
    final_columns = [
        'title', 'price', 'condition', 'model_name', 'storage_gb', 
        'model_year', 'device_age', 'has_defect', 'final_condition_score',
        'shipping_cost'
    ]
    
    df_final = df_clean[final_columns]
    
    # Save as joint_data_collection.csv
    joint_file = os.path.join(data_folder, 'joint_data_collection.csv')
    df_final.to_csv(joint_file, index=False)
    
    print(f"\n✅ Saved joint data collection to: {joint_file}")
    print(f"   Shape: {df_final.shape}")
    print(f"   Columns: {', '.join(df_final.columns.tolist())}")
    
    # Show sample
    print(f"\n📋 Sample of joint_data_collection.csv:")
    print(df_final.head(3).to_string())
    
    return df_final

def split_and_save_data(df):
    """
    Split data into training and test sets, and create activation data
    """
    print("\n" + "="*80)
    print("🔀 SPLITTING DATA")
    print("="*80)
    
    if df is None or len(df) < 10:
        print("❌ Not enough data to split (need at least 10 samples)")
        return
    
    # Features to use for ML (excluding text columns)
    ml_features = [
        'storage_gb', 'device_age', 'has_defect', 
        'final_condition_score', 'shipping_cost'
    ]
    
    # Target variable
    target = 'price'
    
    # Ensure all features exist
    available_features = [col for col in ml_features if col in df.columns]
    print(f"🤖 Using features: {available_features}")
    
    X = df[available_features]
    y = df[target]
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data Split Summary:")
    print(f"   Original dataset size: {len(df)}")
    print(f"   Training set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    
    # Create training DataFrame
    train_df = X_train.copy()
    train_df[target] = y_train
    
    # Add back original columns for context
    train_df['title'] = df.loc[X_train.index, 'title'].values
    train_df['condition'] = df.loc[X_train.index, 'condition'].values
    train_df['model_name'] = df.loc[X_train.index, 'model_name'].values
    
    # Create test DataFrame
    test_df = X_test.copy()
    test_df[target] = y_test
    
    # Add back original columns for context
    test_df['title'] = df.loc[X_test.index, 'title'].values
    test_df['condition'] = df.loc[X_test.index, 'condition'].values
    test_df['model_name'] = df.loc[X_test.index, 'model_name'].values
    
    # Save training and test data
    train_file = os.path.join(data_folder, 'training_data.csv')
    test_file = os.path.join(data_folder, 'test_data.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n💾 Saved training data to: {train_file}")
    print(f"   Training samples: {len(train_df)}")
    print(f"💾 Saved test data to: {test_file}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create activation_data.csv (single entry from test set)
    activation_df = test_df.sample(1, random_state=42)
    activation_file = os.path.join(data_folder, 'activation_data.csv')
    activation_df.to_csv(activation_file, index=False)
    
    print(f"\n🎯 Created activation_data.csv with 1 sample")
    print(f"💾 Saved to: {activation_file}")
    print(f"\n📋 Activation data sample:")
    print(activation_df.to_string())
    
    return train_df, test_df, activation_df

def normalize_data(train_df, test_df):
    """
    Apply algorithmic normalization to numerical features
    """
    print("\n" + "="*80)
    print("📏 APPLYING NORMALIZATION")
    print("="*80)
    
    # Identify numerical columns to normalize
    numerical_cols = ['storage_gb', 'device_age', 'final_condition_score', 'shipping_cost']
    numerical_cols = [col for col in numerical_cols if col in train_df.columns]
    
    print(f"🔢 Normalizing columns: {numerical_cols}")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data only
    train_numerical = train_df[numerical_cols]
    scaler.fit(train_numerical)
    
    # Transform both training and test data
    train_numerical_scaled = scaler.transform(train_numerical)
    test_numerical_scaled = scaler.transform(test_df[numerical_cols])
    
    # Update DataFrames with scaled values
    for i, col in enumerate(numerical_cols):
        train_df[f'{col}_scaled'] = train_numerical_scaled[:, i]
        test_df[f'{col}_scaled'] = test_numerical_scaled[:, i]
    
    print("✅ Normalization completed")
    print(f"   Added scaled columns: {[f'{col}_scaled' for col in numerical_cols]}")
    
    # Save normalized versions
    train_norm_file = os.path.join(data_folder, 'training_data_normalized.csv')
    test_norm_file = os.path.join(data_folder, 'test_data_normalized.csv')
    
    train_df.to_csv(train_norm_file, index=False)
    test_df.to_csv(test_norm_file, index=False)
    
    print(f"\n💾 Saved normalized training data to: {train_norm_file}")
    print(f"💾 Saved normalized test data to: {test_norm_file}")
    
    return train_df, test_df

def get_products_with_details(product_name):
    """
    Extract product titles, prices, and conditions from eBay search results
    """
    # Generate URL based on product name
    url = get_ebay_url(product_name)
    print(f"🔍 Searching for: {product_name}")
    print(f"🌐 URL: {url}")
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Optional: Run in headless mode
    # chrome_options.add_argument("--headless")
    
    # Add user agent to avoid detection
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Initialize Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    
    products = []  # List to store product dictionaries
    
    try:
        print("🚀 Navigating to eBay search page...")
        driver.get(url)
        
        # Wait for page to load completely
        print("⏳ Waiting for page to load...")
        time.sleep(5)
        
        # Scroll to load all products
        print("📜 Scrolling to load all products...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        for i in range(3):  # Scroll 3 times to load more content
            print(f"  Scroll {i+1}/3...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        print("\n🔍 Extracting product details...")
        print("-" * 100)
        
        # Find all product containers
        product_containers = driver.find_elements(By.CSS_SELECTOR, "div.s-card, li.s-item")
        
        if not product_containers:
            # Try alternative selectors
            product_containers = driver.find_elements(By.CSS_SELECTOR, "[data-viewport] li, .srp-results li")
        
        print(f"Found {len(product_containers)} product containers")
        
        for idx, container in enumerate(product_containers[1:], 1):  # Skip first item (often a template)
            try:
                # Extract title
                title = "Not found"
                try:
                    title_element = container.find_element(
                        By.CSS_SELECTOR, 
                        "div.s-card__title span.su-styled-text.primary.default"
                    )
                    title = title_element.text.strip()
                except:
                    # Try alternative title selectors
                    try:
                        title_element = container.find_element(
                            By.CSS_SELECTOR,
                            ".s-item__title span"
                        )
                        title = title_element.text.strip()
                    except:
                        pass
                
                # Extract raw price (with $ sign)
                raw_price = "Not found"
                try:
                    price_element = container.find_element(
                        By.CSS_SELECTOR,
                        "span.su-styled-text.primary.bold.large-1.s-card__price"
                    )
                    raw_price = price_element.text.strip()
                except:
                    # Try alternative price selectors
                    try:
                        price_element = container.find_element(
                            By.CSS_SELECTOR,
                            ".s-item__price"
                        )
                        raw_price = price_element.text.strip()
                    except:
                        pass
                
                # Clean the price (remove $ sign)
                clean_price_value = clean_price(raw_price)
                
                # Extract condition
                condition = "Not specified"
                try:
                    condition_element = container.find_element(
                        By.CSS_SELECTOR,
                        "div.s-card__subtitle span.su-styled-text.secondary.default"
                    )
                    condition = condition_element.text.strip()
                except:
                    # Try alternative condition selectors
                    try:
                        condition_element = container.find_element(
                            By.CSS_SELECTOR,
                            ".s-item__subtitle, .s-item__condition"
                        )
                        condition = condition_element.text.strip()
                    except:
                        # If no condition found, check for refurbished/brand indicators
                        try:
                            # Check if it says "Brand New", "New", etc in title
                            if "brand new" in title.lower() or "new" == title.split()[0].lower():
                                condition = "New"
                            elif "refurbished" in title.lower():
                                condition = "Refurbished"
                            elif "used" in title.lower():
                                condition = "Used"
                        except:
                            pass
                
                # Create product data dictionary with only title, price, condition
                product_data = {
                    "title": title,
                    "price": clean_price_value,  # This is the clean price without $
                    "condition": condition
                }
                products.append(product_data)
                
                # Display progress
                print(f"\n📦 Product {idx:3d}:")
                print(f"   Title: {title[:70]}{'...' if len(title) > 70 else ''}")
                print(f"   Price: {clean_price_value if clean_price_value else 'N/A'}")
                print(f"   Condition: {condition}")
                print(f"   {'─'*40}")
                
                # Stop if we have enough products
                if idx >= 50:
                    print(f"\n⏹️  Stopping at {idx} products")
                    break
                    
            except Exception as e:
                # Skip containers with errors
                continue
        
        # Run the complete data pipeline
        if products:
            print("\n" + "="*80)
            print("🚀 STARTING DATA PROCESSING PIPELINE")
            print("="*80)
            
            # Step 1: Create joint data collection
            joint_df = create_joint_collection(products, product_name)
            
            if joint_df is not None and len(joint_df) > 0:
                # Step 2: Split data
                train_df, test_df, activation_df = split_and_save_data(joint_df)
                
                # Step 3: Normalize data
                if train_df is not None and test_df is not None:
                    normalize_data(train_df, test_df)
                
                # Final summary
                print("\n" + "="*80)
                print("✅ PIPELINE COMPLETE - FILES CREATED")
                print("="*80)
                print(f"📁 All files saved in '{data_folder}' folder:")
                print(f"   1. joint_data_collection.csv - Cleaned dataset with engineered features")
                print(f"   2. training_data.csv - 80% of data for model training")
                print(f"   3. test_data.csv - 20% of data for model testing")
                print(f"   4. activation_data.csv - Single sample for prediction testing")
                print(f"   5. training_data_normalized.csv - Normalized training data")
                print(f"   6. test_data_normalized.csv - Normalized test data")
        
        return products
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        # Close the browser
        print("\n👋 Closing browser...")
        driver.quit()

def quick_extraction(product_name):
    """Quick extraction with minimal features"""
    # Generate URL based on product name
    url = get_ebay_url(product_name)
    print(f"🔍 Quick search for: {product_name}")
    print(f"🌐 URL: {url}")
    
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    products = []
    
    try:
        print("🌐 Loading page...")
        driver.get(url)
        time.sleep(4)
        
        # Scroll to load content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        print("\n🎯 Extracting product details...")
        
        # Direct selectors for all three elements
        products_data = driver.find_elements(By.CSS_SELECTOR, "div.s-card")
        
        for idx, product in enumerate(products_data[:30], 1):  # Limit to 30 products
            try:
                # Get title
                title = product.find_element(
                    By.CSS_SELECTOR,
                    "div.s-card__title span.su-styled-text.primary.default"
                ).text.strip()
                
                # Get raw price
                raw_price = product.find_element(
                    By.CSS_SELECTOR,
                    "span.su-styled-text.primary.bold.large-1.s-card__price"
                ).text.strip()
                
                # Clean the price
                clean_price_value = clean_price(raw_price)
                
                # Get condition
                try:
                    condition = product.find_element(
                        By.CSS_SELECTOR,
                        "div.s-card__subtitle span.su-styled-text.secondary.default"
                    ).text.strip()
                except:
                    condition = "Condition not specified"
                
                products.append({
                    "title": title,
                    "price": clean_price_value,  # Clean price without $
                    "condition": condition
                })
                
                # Display with clean price
                clean_display = clean_price_value if clean_price_value else "N/A"
                print(f"#{idx:2d}: {clean_display:10} | {condition:20} | {title[:50]}{'...' if len(title) > 50 else ''}")
                
            except Exception as e:
                continue
        
        # Run the pipeline for quick extraction too
        if products:
            print("\n🚀 Starting quick data pipeline...")
            joint_df = create_joint_collection(products, product_name)
            if joint_df is not None:
                split_and_save_data(joint_df)
        
        return products
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
    finally:
        driver.quit()

def run_data_pipeline_existing():
    """
    Run the complete data pipeline on existing raw data
    Useful if you already have raw_listings.csv
    """
    print("\n" + "="*80)
    print("🔄 RUNNING PIPELINE ON EXISTING DATA")
    print("="*80)
    
    # Check for existing raw data
    raw_file = os.path.join(data_folder, 'raw_listings.csv')
    if os.path.exists(raw_file):
        print(f"📂 Found existing data: {raw_file}")
        
        # Load raw data
        df_raw = pd.read_csv(raw_file)
        print(f"   Loaded {len(df_raw)} raw listings")
        
        # Convert to products format
        products = []
        for _, row in df_raw.iterrows():
            products.append({
                'title': row['title'],
                'price': row['price'],
                'condition': row['condition']
            })
        
        # Get product name from first entry or default
        product_name = "iPhone"  # Default
        
        # Run the pipeline
        joint_df = create_joint_collection(products, product_name)
        if joint_df is not None:
            train_df, test_df, activation_df = split_and_save_data(joint_df)
            if train_df is not None and test_df is not None:
                normalize_data(train_df, test_df)
        
        return True
    else:
        print(f"❌ No existing raw data found at {raw_file}")
        print("   Please run the scraper first to collect data")
        return False

if __name__ == "__main__":
    print("🛒 SMART PRICE - COMPLETE DATA PIPELINE")
    print("="*80)
    print("📊 This script will:")
    print("   1. Scrape data from eBay")
    print("   2. Clean and engineer features")
    print("   3. Remove outliers algorithmically")
    print("   4. Split data (80/20)")
    print("   5. Create activation data")
    print("   6. Apply algorithmic normalization")
    print("="*80)
    
    print(f"\n📂 Data folder: '{data_folder}'")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"   Created '{data_folder}' folder")
    
    print("\nSelect option:")
    print("1. Run complete pipeline (scrape + process)")
    print("2. Run pipeline on existing data")
    print("3. Quick extraction (limited data)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip() or "1"
    
    if choice == "2":
        run_data_pipeline_existing()
    elif choice == "3":
        product_name = input("Enter product name to search: ").strip() or "iPhone"
        quick_extraction(product_name)
    else:
        # Default to complete pipeline
        product_name = input("\nEnter product name to search (e.g., 'iPhone 13'): ").strip() or "iPhone"
        
        print(f"\n🚀 Starting complete pipeline for: '{product_name}'")
        print("="*80)
        
        products = get_products_with_details(product_name)
        
        # Final verification
        if products:
            print("\n" + "="*80)
            print("🔍 VERIFYING GENERATED FILES")
            print("="*80)
            
            files_to_check = [
                'joint_data_collection.csv',
                'training_data.csv',
                'test_data.csv',
                'activation_data.csv'
            ]
            
            for file in files_to_check:
                file_path = os.path.join(data_folder, file)
                if os.path.exists(file_path):
                    try:
                        df_check = pd.read_csv(file_path)
                        print(f"✅ {file}: {len(df_check)} rows, {len(df_check.columns)} columns")
                    except:
                        print(f"⚠️  {file}: Exists but could not read")
                else:
                    print(f"❌ {file}: Not found")