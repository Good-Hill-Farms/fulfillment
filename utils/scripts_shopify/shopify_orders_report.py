import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Shopify setup
SHOPIFY_ACCESS_TOKEN = os.getenv('SHOPIFY_ACCESS_TOKEN')
SHOP_URL = os.getenv('SHOP_URL')

def update_orders_data(start_date=None, end_date=None):
    """
    Get orders data and process it
    
    Args:
        start_date (datetime, optional): Start date for orders query. Defaults to 90 days ago.
        end_date (datetime, optional): End date for orders query. Defaults to current date.
    Returns:
        pd.DataFrame: DataFrame with order data or empty DataFrame if error occurs
    """
    # Define columns for consistent DataFrame structure
    columns = [
        'Order ID', 'Created At', 'Subtotal', 'Shipping', 'Total',
        'Discount Code', 'Discount Amount', 'Delivery Date', 'Shipping Method',
        'Quantity', 'SKU', 'Unit Price', 'Tags', 'Customer Type'
    ]
    
    try:
        headers = {
            'X-Shopify-Access-Token': SHOPIFY_ACCESS_TOKEN,
            'Content-Type': 'application/json'
        }
        
        # Get date range - default to last 90 days if not specified
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=90)
            
        date_query = f'created_at:>={start_date.strftime("%Y-%m-%d")} AND created_at:<={end_date.strftime("%Y-%m-%d")} AND NOT tag:replacement AND total_price:>0'
        
        logger.info(f"Fetching orders from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_orders = []
        has_next_page = True
        cursor = None
        
        while has_next_page:
            # Build pagination part of the query
            after_cursor = f', after: "{cursor}"' if cursor else ''
            
            query = f'''
            {{
                orders(first: 250, sortKey: CREATED_AT, reverse: true, query: "{date_query}"{after_cursor}) {{
                    edges {{
                        cursor
                        node {{
                            id
                            name
                            createdAt
                            tags
                            subtotalPriceSet {{
                                shopMoney {{
                                    amount
                                }}
                            }}
                            totalShippingPriceSet {{
                                shopMoney {{
                                    amount
                                }}
                            }}
                            totalPriceSet {{
                                shopMoney {{
                                    amount
                                }}
                            }}
                            discountCode
                            discountApplications(first: 10) {{
                                edges {{
                                    node {{
                                        ... on DiscountCodeApplication {{
                                            code
                                            targetType
                                            value {{
                                                ... on MoneyV2 {{
                                                    amount
                                                }}
                                                ... on PricingPercentageValue {{
                                                    percentage
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                            shippingLine {{
                                title
                                carrierIdentifier
                                source
                                code
                            }}
                            customer {{
                                firstName
                                lastName
                                email
                                tags
                            }}
                            lineItems(first: 50) {{
                                edges {{
                                    node {{
                                        quantity
                                        sku
                                        originalUnitPriceSet {{
                                            shopMoney {{
                                                amount
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                    pageInfo {{
                        hasNextPage
                        endCursor
                    }}
                }}
            }}
            '''
            
            try:
                response = requests.post(
                    f'https://{SHOP_URL}/admin/api/2024-01/graphql.json',
                    headers=headers,
                    json={'query': query}
                )
                
                data = response.json()
                
                if not data or 'data' not in data or 'orders' not in data['data']:
                    logger.warning("Invalid API response structure")
                    return pd.DataFrame(columns=columns)
                
                if 'errors' in data:
                    logger.warning("\nAPI Errors:", data['errors'])
                    return pd.DataFrame(columns=columns)
                
                orders = data['data']['orders']['edges']
                if orders:
                    all_orders.extend(orders)
                    page_info = data['data']['orders'].get('pageInfo', {})
                    has_next_page = page_info.get('hasNextPage', False)
                    cursor = page_info.get('endCursor') if has_next_page else None
                else:
                    has_next_page = False
                    
            except Exception as e:
                logger.error(f"Error in API call: {str(e)}")
                return pd.DataFrame(columns=columns)
        
        if not all_orders:
            logger.warning("No orders found")
            return pd.DataFrame(columns=columns)
                    
        rows = []
        for order in all_orders:
            try:
                node = order.get('node', {})
                if not node:
                    continue
                    
                total_price_set = node.get('totalPriceSet', {}).get('shopMoney', {})
                if not total_price_set or 'amount' not in total_price_set:
                    continue
                    
                total_price = float(total_price_set['amount'])
                if total_price <= 0:
                    continue
                
                order_id = node.get('id', '').split('/')[-1]
                created_at = datetime.fromisoformat(node.get('createdAt', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                
                subtotal = float(node.get('subtotalPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                shipping = float(node.get('totalShippingPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                
                shipping_info = node.get('shippingLine') or {}
                delivery_date = shipping_info.get('title', '')
                shipping_method = shipping_info.get('source') or shipping_info.get('code') or shipping_info.get('carrierIdentifier', '')
                
                line_items = node.get('lineItems', {}).get('edges', [])
                for item in line_items:
                    line_item = item.get('node', {})
                    if not line_item:
                        continue
                        
                    sku = line_item.get('sku') or 'N/A'
                    if sku in ['e.ripening_guide', 'monthly-priority-pass', 'N/A']:
                        continue
                    
                    discount_code = ''
                    discount_amount = 0.0
                    discount_apps = node.get('discountApplications', {}).get('edges', [])
                    if discount_apps:
                        discount_app = discount_apps[0].get('node', {})
                        if discount_app:
                            discount_code = discount_app.get('code', '')
                            discount_value = discount_app.get('value', {})
                            if 'percentage' in discount_value:
                                percentage = float(discount_value['percentage'])
                                original_price = float(line_item.get('originalUnitPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                                discount_amount = (original_price * percentage / 100) * line_item.get('quantity', 0)
                            elif 'amount' in discount_value:
                                discount_amount = float(discount_value['amount'])
                    
                    rows.append([
                        order_id,
                        created_at,
                        subtotal,
                        shipping,
                        total_price,
                        discount_code,
                        discount_amount,
                        delivery_date,
                        shipping_method,
                        line_item.get('quantity', 0),
                        sku,
                        float(line_item.get('originalUnitPriceSet', {}).get('shopMoney', {}).get('amount', 0)),
                        ', '.join(node.get('tags', [])),
                        'Returning' if 'Returning Buyer' in node.get('tags', []) else 'New'
                    ])
            except Exception as e:
                logger.info(f"Error processing order: {str(e)}")
                continue
        
        if not rows:
            logger.warning("No valid orders to process")
            return pd.DataFrame(columns=columns)
            
        df = pd.DataFrame(rows, columns=columns)
        return df
        
    except Exception as e:
        logger.error(f"\n! Error updating Orders report: {str(e)}")
        return pd.DataFrame(columns=columns)

if __name__ == '__main__':
    update_orders_data(start_date=datetime(2025, 7, 1), end_date=datetime(2025, 7, 9)) 