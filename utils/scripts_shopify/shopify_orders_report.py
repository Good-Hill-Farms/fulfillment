import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Import SKU mapping functions
try:
    # Try relative import first (when running from project root)
    from utils.google_sheets import load_sku_mappings_from_sheets, load_sku_type_data
except ImportError:
    try:
        # Try absolute import (when running from utils/scripts_shopify/)
        import sys
        import os
        # Add the project root to the path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from utils.google_sheets import load_sku_mappings_from_sheets, load_sku_type_data
    except ImportError:
        logger.warning("Could not import SKU mapping functions")
        load_sku_mappings_from_sheets = None
        load_sku_type_data = None

# Shopify setup
SHOPIFY_ACCESS_TOKEN = os.getenv('SHOPIFY_ACCESS_TOKEN')
SHOP_URL = os.getenv('SHOP_URL')

def get_product_type_from_sku(sku, sku_mappings=None, sku_type_df=None, failed_skus=None):
    """
    Get product type from SKU using the SKU mappings with fallback to SKU type data.
    
    Args:
        sku (str): The SKU to lookup
        sku_mappings (dict, optional): SKU mappings dictionary
        sku_type_df (pd.DataFrame, optional): SKU type data for fallback
        failed_skus (set, optional): Set to track SKUs where product type was not found
    
    Returns:
        str: Product type or 'Unknown' if not found
    """
    if not sku or pd.isna(sku):
        return "Unknown"
    
    # Load SKU mappings if not provided
    if sku_mappings is None:
        if load_sku_mappings_from_sheets:
            try:
                sku_mappings = load_sku_mappings_from_sheets()
            except Exception as e:
                logger.debug(f"Could not load SKU mappings: {e}")
                sku_mappings = None
        else:
            sku_mappings = None
    
    # Try SKU mappings first
    if sku_mappings and isinstance(sku_mappings, dict):
        # Check SKU mappings for both warehouses
        for location in ['Oxnard', 'Wheeling']:
            if location in sku_mappings and 'singles' in sku_mappings[location]:
                if sku in sku_mappings[location]['singles']:
                    product_type = sku_mappings[location]['singles'][sku].get('pick_type_inventory', 'Unknown')
                    if product_type and product_type != 'Unknown':
                        return product_type
            if location in sku_mappings and 'bundles' in sku_mappings[location]:
                if sku in sku_mappings[location]['bundles']:
                    # For bundles, use the first component's product type
                    components = sku_mappings[location]['bundles'][sku]
                    if components and len(components) > 0:
                        product_type = components[0].get('pick_type', 'Unknown')
                        if product_type and product_type != 'Unknown':
                            return product_type
    
    # Fallback to SKU type data
    if sku_type_df is None and load_sku_type_data:
        try:
            sku_type_df = load_sku_type_data()
        except Exception as e:
            logger.debug(f"Could not load SKU type data: {e}")
            sku_type_df = None
    
    if sku_type_df is not None and not sku_type_df.empty:
        # Look for the SKU in the type data
        sku_row = sku_type_df[sku_type_df['SKU'] == sku]
        if not sku_row.empty:
            product_type = sku_row.iloc[0]['PRODUCT_TYPE']
            if product_type and str(product_type).strip() and str(product_type).strip() != 'Unknown':
                return str(product_type).strip()
    
    # Track failed SKU lookup
    if failed_skus is not None and sku and str(sku).strip():
        failed_skus.add(str(sku).strip())
    
    return "Unknown"

def update_fulfilled_orders_data(start_date=None, end_date=None):
    """
    Get fulfilled orders data using simplified query
    
    Args:
        start_date (datetime, optional): Start date for orders query. Defaults to 90 days ago.
        end_date (datetime, optional): End date for orders query. Defaults to current date.
    Returns:
        pd.DataFrame: DataFrame with fulfilled order data or empty DataFrame if error occurs
    """
    # Define columns for consistent DataFrame structure (removed customer data)
    columns = [
        'Order ID', 'Order Name', 'Created At', 'Subtotal', 'Shipping', 'Total',
        'Discount Code', 'Discount Amount', 'Delivery Date', 
        'Quantity', 'SKU', 'Product Type', 'Unit Price', 'Tags'
    ]
    
    # Load SKU mappings for product type lookup
    sku_mappings = None
    if load_sku_mappings_from_sheets:
        try:
            sku_mappings = load_sku_mappings_from_sheets()
        except Exception as e:
            logger.debug(f"Could not load SKU mappings: {e}")
    
    # Load SKU type data as fallback
    sku_type_df = None
    if load_sku_type_data:
        try:
            sku_type_df = load_sku_type_data()
        except Exception as e:
            logger.debug(f"Could not load SKU type data: {e}")
    
    # Track SKUs where product type was not found
    failed_skus = set()
    
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
        
        logger.debug(f"Fetching fulfilled orders from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_orders = []
        has_next_page = True
        cursor = None
        
        while has_next_page:
            # Build pagination part of the query
            after_cursor = f', after: "{cursor}"' if cursor else ''
            
            # Escape quotes in the date query
            date_query_escaped = date_query.replace('"', '\\"')
            
            query = f'''
            query getOrders {{
                orders(first: 250, sortKey: CREATED_AT, reverse: true, query: "{date_query_escaped}"{after_cursor}) {{
                    edges {{
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
                order_name = node.get('name', '')
                created_at = datetime.fromisoformat(node.get('createdAt', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                
                subtotal = float(node.get('subtotalPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                shipping = float(node.get('totalShippingPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                
                shipping_info = node.get('shippingLine') or {}
                delivery_date = shipping_info.get('title', '')
                
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
                    
                    unit_price = float(line_item.get('originalUnitPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                    quantity = line_item.get('quantity', 0)
                    
                    rows.append([
                        order_id,
                        order_name,
                        created_at,
                        subtotal,
                        shipping,
                        total_price,
                        discount_code,
                        discount_amount,
                        delivery_date,
                        quantity,
                        sku,
                        get_product_type_from_sku(sku, sku_mappings, sku_type_df, failed_skus),
                        unit_price,
                        ', '.join(node.get('tags', []))
                    ])
            except Exception as e:
                logger.info(f"Error processing order: {str(e)}")
                continue
        
        if not rows:
            logger.warning("No valid orders to process")
            return pd.DataFrame(columns=columns)
            
        df = pd.DataFrame(rows, columns=columns)
        
        # Log summary of SKUs without product type
        if failed_skus:
            logger.info(f"\n📊 FULFILLED ORDERS - SKUs without Product Type found ({len(failed_skus)} unique SKUs):")
            for sku in sorted(failed_skus):
                logger.info(f"  - {sku}")
            logger.info(f"📊 These SKUs appear in {len(df[df['Product Type'] == 'Unknown'])} order line items")
        else:
            logger.info("✅ FULFILLED ORDERS - All SKUs have Product Type mappings!")
        
        return df
        
    except Exception as e:
        logger.error(f"\n! Error updating fulfilled orders report: {str(e)}")
        return pd.DataFrame(columns=columns)

def update_orders_data(start_date=None, end_date=None):
    """
    Get orders data and process it (updated to use simplified query)
    
    Args:
        start_date (datetime, optional): Start date for orders query. Defaults to 90 days ago.
        end_date (datetime, optional): End date for orders query. Defaults to current date.
    Returns:
        pd.DataFrame: DataFrame with order data or empty DataFrame if error occurs
    """
    # Define columns for consistent DataFrame structure
    columns = [
        'Order ID', 'Created At', 'Fulfillment Status', 'Subtotal', 'Shipping', 'Line Item Total',
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
        
        logger.debug(f"Fetching orders from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_orders = []
        has_next_page = True
        cursor = None
        
        while has_next_page:
            # Build pagination part of the query
            after_cursor = f', after: "{cursor}"' if cursor else ''
            
            # Escape quotes in the date query
            date_query_escaped = date_query.replace('"', '\\"')
            
            query = f'''
            query getOrders {{
                orders(first: 250, sortKey: CREATED_AT, reverse: true, query: "{date_query_escaped}"{after_cursor}) {{
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
                    
                    # Calculate line item total (unit price * quantity - discount)
                    unit_price = float(line_item.get('originalUnitPriceSet', {}).get('shopMoney', {}).get('amount', 0))
                    quantity = line_item.get('quantity', 0)
                    line_item_total = (unit_price * quantity) - discount_amount
                    
                    rows.append([
                        order_id,
                        created_at,
                        node.get('displayFulfillmentStatus', ''),
                        subtotal,
                        shipping,
                        line_item_total,  # Use line item total instead of full order total
                        discount_code,
                        discount_amount,
                        delivery_date,
                        shipping_method,
                        quantity,
                        sku,
                        unit_price,
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

def update_unfulfilled_orders(start_date=None, end_date=None, all_unfulfilled=False):
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
        'Order ID', 'Created At', 'Order Name', 'SKU', 'Product Type', 'Product Name',
        'Unfulfilled Quantity', 'Unit Price', 'Line Item Total',
        'Delivery Date', 'Shipping Method', 'Tags'
    ]
    
    # Load SKU mappings for product type lookup
    sku_mappings = None
    if load_sku_mappings_from_sheets:
        try:
            sku_mappings = load_sku_mappings_from_sheets()
        except Exception as e:
            logger.debug(f"Could not load SKU mappings: {e}")
    
    # Load SKU type data as fallback
    sku_type_df = None
    if load_sku_type_data:
        try:
            sku_type_df = load_sku_type_data()
        except Exception as e:
            logger.debug(f"Could not load SKU type data: {e}")
    
    # Track SKUs where product type was not found
    failed_skus = set()
    
    try:
        headers = {
            'X-Shopify-Access-Token': SHOPIFY_ACCESS_TOKEN,
            'Content-Type': 'application/json'
        }
        
        if all_unfulfilled:
            # Get all unfulfilled orders regardless of creation date
            date_query = "financial_status:paid NOT cancelled:true (fulfillment_status:unfulfilled OR fulfillment_status:partial)"
            logger.debug("Fetching all unfulfilled orders (no date filter)")
        else:
            # Get date range - default to last 90 days if not specified
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
                
            date_query = f"created_at:>='{start_date.strftime('%Y-%m-%d')}' created_at:<='{end_date.strftime('%Y-%m-%d')}' financial_status:paid NOT cancelled:true (fulfillment_status:unfulfilled OR fulfillment_status:partial)"
            
            logger.debug(f"Fetching unfulfilled orders from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
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
                    line_item_total = unit_price * unfulfilled_quantity
                    product_name = line_item.get('name', '')
                    
                    rows.append([
                        order_id,
                        created_at,
                        order_name,
                        sku,
                        get_product_type_from_sku(sku, sku_mappings, sku_type_df, failed_skus),
                        product_name,
                        unfulfilled_quantity,
                        unit_price,
                        line_item_total,
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
        
        # Log summary of SKUs without product type
        if failed_skus:
            logger.info(f"\n📊 UNFULFILLED ORDERS - SKUs without Product Type found ({len(failed_skus)} unique SKUs):")
            for sku in sorted(failed_skus):
                logger.info(f"  - {sku}")
            logger.info(f"📊 These SKUs appear in {len(df[df['Product Type'] == 'Unknown'])} order line items")
        else:
            logger.info("✅ UNFULFILLED ORDERS - All SKUs have Product Type mappings!")
        
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
    
    # Get orders data (detailed query with fulfillment info)
    orders_df = update_orders_data(start_date=monday, end_date=today)
    
    # Get fulfilled orders data (simplified query with customer info)
    fulfilled_orders_df = update_fulfilled_orders_data(start_date=monday, end_date=today)
    
    # Get unfulfilled orders
    unfulfilled_df = update_unfulfilled_orders(start_date=monday, end_date=today)
    
    # Save to CSV files in sheet_data directory
    if not orders_df.empty:
        orders_filename = os.path.join(output_dir, f"orders_report_{date_str}.csv")
        orders_df.to_csv(orders_filename, index=False)
        logger.info(f"Orders report (detailed) saved to {orders_filename}")
    
    if not fulfilled_orders_df.empty:
        fulfilled_filename = os.path.join(output_dir, f"fulfilled_orders_report_{date_str}.csv")
        fulfilled_orders_df.to_csv(fulfilled_filename, index=False)
        logger.info(f"Fulfilled orders report (simplified) saved to {fulfilled_filename}")
    
    if not unfulfilled_df.empty:
        unfulfilled_filename = os.path.join(output_dir, f"unfulfilled_orders_{date_str}.csv")
        unfulfilled_df.to_csv(unfulfilled_filename, index=False)
        logger.info(f"Unfulfilled orders report saved to {unfulfilled_filename}")
        
    # Print summary of saved files
    print("\nShopify Orders Reports saved to the 'sheet_data' directory!")
    print("Summary of files saved:")
    print("------------------------")
    for filename in [f"orders_report_{date_str}.csv", f"fulfilled_orders_report_{date_str}.csv", f"unfulfilled_orders_{date_str}.csv"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # Convert to KB
            print(f"{filename:<35} {size:.1f} KB") 