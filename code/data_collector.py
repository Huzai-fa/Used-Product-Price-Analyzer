# Save this as data_collector.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

class DataCollector:
    def __init__(self):
        self.data = []
        
    def scrape_ebay(self, product_name, max_pages=2):
        """Scrape eBay for used product listings"""
        print(f"Scraping eBay for: {product_name}")
        
        for page in range(1, max_pages + 1):
            url = f"https://www.ebay.com/sch/i.html?_nkw={product_name.replace(' ', '+')}&_pgn={page}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all product listings
                items = soup.find_all('div', class_='s-item__info')
                
                for item in items:
                    try:
                        # Extract title
                        title_elem = item.find('h3', class_='s-item__title')
                        title = title_elem.text if title_elem else 'N/A'
                        
                        # Extract price
                        price_elem = item.find('span', class_='s-item__price')
                        price_text = price_elem.text if price_elem else '$0'
                        price = self._clean_price(price_text)
                        
                        # Extract condition
                        condition_elem = item.find('span', class_='SECONDARY_INFO')
                        condition = condition_elem.text if condition_elem else 'Used'
                        
                        # Extract shipping
                        shipping_elem = item.find('span', class_='s-item__shipping')
                        shipping = shipping_elem.text if shipping_elem else 'Free shipping'
                        shipping_cost = self._clean_shipping(shipping)
                        
                        if price > 0:  # Only add valid listings
                            self.data.append({
                                'title': title,
                                'price': price,
                                'condition': condition,
                                'shipping_cost': shipping_cost,
                                'source': 'ebay',
                                'product_category': product_name
                            })
                            
                    except Exception as e:
                        continue
                
                print(f"  Page {page}: Found {len(items)} items")
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
        
        return pd.DataFrame(self.data)
    
    def _clean_price(self, price_str):
        """Convert price string to float"""
        try:
            # Handle multiple prices (e.g., "C $123.45 to C $234.56")
            numbers = re.findall(r'\d+\.\d+', price_str)
            if numbers:
                return float(numbers[0])
            return 0.0
        except:
            return 0.0
    
    def _clean_shipping(self, shipping_str):
        """Extract shipping cost"""
        if 'free' in shipping_str.lower():
            return 0.0
        try:
            numbers = re.findall(r'\d+\.\d+', shipping_str)
            if numbers:
                return float(numbers[0])
            return 0.0
        except:
            return 0.0

# Test function
def test_collector():
    collector = DataCollector()
    df = collector.scrape_ebay("iphone 12", max_pages=2)
    print(f"Collected {len(df)} listings")
    if len(df) > 0:
        print(df.head())
    return df

if __name__ == "__main__":
    test_collector()