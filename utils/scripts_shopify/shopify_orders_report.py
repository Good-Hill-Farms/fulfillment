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
        'Order ID', 'Created At', 'Fulfillment Status', 'Subtotal', 'Shipping', 'Total',
        'Discount Code', 'Discount Amount', 'Delivery Date', 'Shipping Method',
        'Quantity', 'SKU', 'Unit Price', 'Tags', 'Customer Type', 'Fulfilled At', 'Fulfillment Duration'
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
            
            # Escape quotes in the date query
            date_query = date_query.replace('"', '\\"')
            
            query = f'''
            query getOrders {{
                orders(first: 250, sortKey: CREATED_AT, reverse: true, query: "{date_query}"{after_cursor}) {{
                    edges {{
                        node {{
                            id
                            name
                            createdAt
                            displayFulfillmentStatus
                            tags
                            fulfillments(first: 5) {{
                                createdAt
                                status
                                trackingInfo {{
                                    company
                                    number
                                }}
                            }}
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
                
                # Add debug logging
                logger.info(f"API Response: {data}")
                
                if not data or 'data' not in data:
                    logger.warning(f"Invalid API response structure. Response: {data}")
                    return pd.DataFrame(columns=columns)
                
                if 'errors' in data:
                    logger.warning(f"\nAPI Errors: {data['errors']}")
                    return pd.DataFrame(columns=columns)
                
                if 'data' not in data or 'orders' not in data['data']:
                    logger.warning(f"Missing orders data. Response structure: {data}")
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
                
                # Get fulfillment information
                fulfilled_at = None
                fulfillment_duration = None
                fulfillments = node.get('fulfillments', [])
                display_status = node.get('displayFulfillmentStatus')
                
                if fulfillments:
                    # Sort fulfillments by creation date to get the first one
                    fulfillments.sort(key=lambda x: x.get('createdAt', ''))
                    first_fulfillment = fulfillments[0]
                    
                    if first_fulfillment and first_fulfillment.get('status') == 'SUCCESS':
                        fulfilled_at = datetime.fromisoformat(first_fulfillment.get('createdAt', '').replace('Z', '+00:00'))
                        created_dt = datetime.fromisoformat(node.get('createdAt', '').replace('Z', '+00:00'))
                        
                        # Calculate duration
                        duration = fulfilled_at - created_dt
                        duration_minutes = duration.total_seconds() / 60
                        
                        # Format the fulfillment time and duration
                        fulfilled_at = fulfilled_at.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Format duration based on length
                        if duration_minutes < 60:  # Less than an hour
                            fulfillment_duration = f"{int(duration_minutes)} min"
                        elif duration_minutes < 24 * 60:  # Less than a day
                            hours = duration_minutes / 60
                            fulfillment_duration = f"{int(hours)} hours"
                        else:  # Days
                            days = duration_minutes / (24 * 60)
                            fulfillment_duration = f"{int(days)} days"
                            
                        
                        # Get tracking info if available
                        tracking_info = first_fulfillment.get('trackingInfo', [])
                        if tracking_info:
                            tracking = tracking_info[0]
                            logger.info(f"Tracking: {tracking.get('company')} - {tracking.get('number')}")
                
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
                        node.get('displayFulfillmentStatus', ''),
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
                        'Returning' if 'Returning Buyer' in node.get('tags', []) else 'New',
                        fulfilled_at if fulfilled_at else '',
                        str(fulfillment_duration) if fulfillment_duration is not None else ''
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

def update_unfulfilled_orders(start_date=None, end_date=None):
    """
    Get unfulfilled orders data
    
    Args:
        start_date (datetime, optional): Start date for orders query. Defaults to 90 days ago.
        end_date (datetime, optional): End date for orders query. Defaults to current date.
    Returns:
        pd.DataFrame: DataFrame with unfulfilled order data or empty DataFrame if error occurs
    """
    # Define columns for consistent DataFrame structure
    columns = [
        'Order ID', 'Created At', 'Order Name', 'SKU', 'Product Name',
        'Unfulfilled Quantity', 'Unit Price', 'Total Price',
        'Delivery Date', 'Shipping Method', 'Tags'
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
            
        date_query = f"created_at:>='{start_date.strftime('%Y-%m-%d')}' created_at:<='{end_date.strftime('%Y-%m-%d')}' financial_status:paid NOT cancelled:true (fulfillment_status:unfulfilled OR fulfillment_status:partial)"
        
        logger.info(f"Fetching unfulfilled orders from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_orders = []
        has_next_page = True
        cursor = None
        
        while has_next_page:
            # Build pagination part of the query
            after_cursor = f', after: "{cursor}"' if cursor else ''
            
            query = f'''
            {{
                orders(first: 250, query: "{date_query}"{after_cursor}) {{
                    edges {{
                        node {{
                            id
                            name
                            createdAt
                            displayFulfillmentStatus
                            tags
                            shippingLine {{
                                title
                                carrierIdentifier
                                source
                                code
                            }}
                            lineItems(first: 50) {{
                                edges {{
                                    node {{
                                        sku
                                        name
                                        quantity
                                        unfulfilledQuantity
                                        fulfillmentStatus
                                        originalUnitPriceSet {{
                                            shopMoney {{
                                                amount
                                            }}
                                        }}
                                        product {{
                                            title
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
                
                # Add debug logging
                logger.info(f"API Response: {data}")
                
                if not data or 'data' not in data:
                    logger.warning(f"Invalid API response structure. Response: {data}")
                    return pd.DataFrame(columns=columns)
                
                if 'errors' in data:
                    logger.warning(f"\nAPI Errors: {data['errors']}")
                    return pd.DataFrame(columns=columns)
                
                if 'data' not in data or 'orders' not in data['data']:
                    logger.warning(f"Missing orders data. Response structure: {data}")
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
            logger.warning("No unfulfilled orders found")
            return pd.DataFrame(columns=columns)
                    
        rows = []
        for order in all_orders:
            try:
                node = order.get('node', {})
                if not node:
                    continue
                
                order_id = node.get('id', '').split('/')[-1]
                order_name = node.get('name', '')
                created_at = datetime.fromisoformat(node.get('createdAt', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                
                shipping_info = node.get('shippingLine') or {}
                delivery_date = shipping_info.get('title', '')
                shipping_method = shipping_info.get('source') or shipping_info.get('code') or shipping_info.get('carrierIdentifier', '')
                
                tags = ', '.join(node.get('tags', []))
                
                line_items = node.get('lineItems', {}).get('edges', [])
                for item in line_items:
                    line_item = item.get('node', {})
                    if not line_item:
                        continue
                    
                    sku = line_item.get('sku') or 'N/A'
                    if sku in ['e.ripening_guide', 'monthly-priority-pass', 'N/A']:
                        continue
                    
                    unfulfilled_quantity = line_item.get('unfulfilledQuantity', 0)
                    if unfulfilled_quantity <= 0:
                        continue
                        
                    unit_price = float(line_item.get('originalUnitPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                    total_price = unit_price * unfulfilled_quantity
                    product_name = line_item.get('name', '')
                    
                    rows.append([
                        order_id,
                        created_at,
                        order_name,
                        sku,
                        product_name,
                        unfulfilled_quantity,
                        unit_price,
                        total_price,
                        delivery_date,
                        shipping_method,
                        tags
                    ])
                    
            except Exception as e:
                logger.info(f"Error processing order: {str(e)}")
                continue
        
        if not rows:
            logger.warning("No valid unfulfilled orders to process")
            return pd.DataFrame(columns=columns)
            
        df = pd.DataFrame(rows, columns=columns)
        return df
        
    except Exception as e:
        logger.error(f"\n! Error updating Unfulfilled Orders report: {str(e)}")
        return pd.DataFrame(columns=columns)

if __name__ == '__main__':
    # Calculate date range from Monday to today
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Format dates for filename
    date_str = f"{monday.strftime('%Y%m%d')}-{today.strftime('%Y%m%d')}"
    
    # Create sheet_data directory if it doesn't exist
    output_dir = "sheet_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get orders data
    orders_df = update_orders_data(start_date=monday, end_date=today)
    unfulfilled_df = update_unfulfilled_orders(start_date=monday, end_date=today)
    
    # Save to CSV files in sheet_data directory
    if not orders_df.empty:
        orders_filename = os.path.join(output_dir, f"orders_report_{date_str}.csv")
        orders_df.to_csv(orders_filename, index=False)
        logger.info(f"Orders report saved to {orders_filename}")
    
    if not unfulfilled_df.empty:
        unfulfilled_filename = os.path.join(output_dir, f"unfulfilled_orders_{date_str}.csv")
        unfulfilled_df.to_csv(unfulfilled_filename, index=False)
        logger.info(f"Unfulfilled orders report saved to {unfulfilled_filename}")
        
    # Print summary of saved files
    print("\nShopify Orders Reports saved to the 'sheet_data' directory!")
    print("Summary of files saved:")
    print("------------------------")
    for filename in [f"orders_report_{date_str}.csv", f"unfulfilled_orders_{date_str}.csv"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # Convert to KB
            print(f"{filename:<30} {size:.1f} KB") 