import requests
import os
from dotenv import load_dotenv
import pandas as pd
import logging
from datetime import datetime, timedelta

load_dotenv()
logger = logging.getLogger(__name__)

# Shopify setup
SHOPIFY_ACCESS_TOKEN = os.getenv('SHOPIFY_ACCESS_TOKEN')
SHOP_URL = os.getenv('SHOP_URL')

def get_priority_unfulfilled_orders():
    """
    Get priority unfulfilled orders from the last 6 months with P1 tags, "a lot of fruit" tags, or orders over $200
    
    Returns:
        pd.DataFrame: DataFrame with priority unfulfilled order data
    """
    # Define columns for the DataFrame
    columns = [
        'Order ID', 'Order Name', 'Created At', 'Tags', 'Total Price', 'Currency',
        'Fulfillment Status', 'Product Name', 'SKU', 'Quantity', 'Unfulfilled Quantity'
    ]
    
    try:
        headers = {
            'X-Shopify-Access-Token': SHOPIFY_ACCESS_TOKEN,
            'Content-Type': 'application/json'
        }
        
        # Calculate date for 6 months ago (exactly half a year)
        today = datetime.now()
        if today.month <= 6:
            six_months_ago = today.replace(year=today.year - 1, month=today.month + 6)
        else:
            six_months_ago = today.replace(month=today.month - 6)
        date_filter = six_months_ago.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching priority unfulfilled orders from {date_filter}...")
        
        # Create the query with date filter for last 6 months and exclude unwanted SKUs
        query_string = f"fulfillment_status:unfulfilled AND created_at:>={date_filter} AND (tag:P1 OR tag:'a lot of fruit' OR total_price:>200) AND NOT sku:e.ripening_guide AND NOT sku:monthly-priority-pass"
        
        query = '''
        {
          orders(first: 50, query: "''' + query_string + '''") {
            edges {
              node {
                id
                name
                createdAt
                tags
                totalPriceSet {
                  shopMoney {
                    amount
                    currencyCode
                  }
                }
                displayFulfillmentStatus
                lineItems(first: 10) {
                  edges {
                    node {
                      name
                      sku
                      quantity
                      unfulfilledQuantity
                    }
                  }
                }
              }
            }
          }
        }
        '''
        
        response = requests.post(
            f'https://{SHOP_URL}/admin/api/2024-01/graphql.json',
            headers=headers,
            json={'query': query}
        )
        
        data = response.json()
        
        if not data or 'data' not in data:
            logger.warning(f"Invalid API response structure. Response: {data}")
            return pd.DataFrame(columns=columns)
        
        if 'errors' in data:
            logger.warning(f"API Errors: {data['errors']}")
            return pd.DataFrame(columns=columns)
        
        if 'data' not in data or 'orders' not in data['data']:
            logger.warning(f"Missing orders data. Response structure: {data}")
            return pd.DataFrame(columns=columns)
        
        orders = data['data']['orders']['edges']
        
        if not orders:
            logger.info("No priority unfulfilled orders found")
            return pd.DataFrame(columns=columns)
        
        rows = []
        for order in orders:
            try:
                node = order.get('node', {})
                if not node:
                    continue
                
                order_id = node.get('id', '').split('/')[-1]
                order_name = node.get('name', '')
                created_at = datetime.fromisoformat(node.get('createdAt', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                tags = ', '.join(node.get('tags', []))
                
                total_price_info = node.get('totalPriceSet', {}).get('shopMoney', {})
                total_price = float(total_price_info.get('amount', 0))
                currency = total_price_info.get('currencyCode', 'USD')
                
                fulfillment_status = node.get('displayFulfillmentStatus', '')
                
                line_items = node.get('lineItems', {}).get('edges', [])
                
                if not line_items:
                    # Add order without line items if no line items exist
                    rows.append([
                        order_id,
                        order_name,
                        created_at,
                        tags,
                        total_price,
                        currency,
                        fulfillment_status,
                        '',    # Product Name
                        'N/A', # SKU
                        0,     # Quantity
                        0      # Unfulfilled Quantity
                    ])
                else:
                    for item in line_items:
                        line_item = item.get('node', {})
                        if not line_item:
                            continue
                        
                        product_name = line_item.get('name', '')
                        sku = line_item.get('sku') or 'N/A'
                        quantity = line_item.get('quantity', 0)
                        unfulfilled_quantity = line_item.get('unfulfilledQuantity', 0)
                        
                        # Only include line items that actually need fulfillment
                        # This ensures we only save items that are truly unfulfilled
                        if unfulfilled_quantity <= 0:
                            continue
                        
                        rows.append([
                            order_id,
                            order_name,
                            created_at,
                            tags,
                            total_price,
                            currency,
                            fulfillment_status,
                            product_name,
                            sku,
                            quantity,
                            unfulfilled_quantity
                        ])
                        
            except Exception as e:
                logger.warning(f"Error processing order: {str(e)}")
                continue
        
        if not rows:
            logger.info("No valid priority unfulfilled orders to process")
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(rows, columns=columns)
        
        logger.info(f"Found {len(df)} priority unfulfilled order line items across {df['Order ID'].nunique()} orders")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching priority unfulfilled orders: {str(e)}")
        return pd.DataFrame(columns=columns)

def save_priority_unfulfilled_orders():
    """
    Fetch and save priority unfulfilled orders to CSV
    """
    # Get the data
    df = get_priority_unfulfilled_orders()
    
    if df.empty:
        print("No priority unfulfilled orders found.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = "sheet_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with current date
    today = datetime.now()
    date_str = today.strftime('%Y%m%d')
    filename = os.path.join(output_dir, f"priority_unfulfilled_orders_{date_str}.csv")
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    # Print summary
    print(f"\nPriority Unfulfilled Orders Report saved to: {filename}")
    print("Summary:")
    print(f"  Total line items: {len(df)}")
    print(f"  Unique orders: {df['Order ID'].nunique()}")
    
    # Show breakdown by criteria
    p1_orders = df[df['Tags'].str.contains('P1', na=False)]['Order ID'].nunique()
    fruit_orders = df[df['Tags'].str.contains('a lot of fruit', na=False)]['Order ID'].nunique()
    high_value_orders = df[df['Total Price'] > 200]['Order ID'].nunique()
    
    print(f"  Orders with P1 tag: {p1_orders}")
    print(f"  Orders with 'a lot of fruit' tag: {fruit_orders}")
    print(f"  Orders over $200: {high_value_orders}")
    
    return df

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the script
    save_priority_unfulfilled_orders() 