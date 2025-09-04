import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from utils.google_sheets import get_credentials
from utils.scripts_shopify.shopify_orders_report import update_fulfilled_orders_data
from utils.data_processor import DataProcessor

st.set_page_config(layout="wide")

# Define required scopes
SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/bigquery',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets'
]

# Add header and description
st.title("Order Fulfillment")
st.caption("Shopify orders and ColdCart shipment details for the selected date range")
st.markdown("""
This page helps track fruit fulfillment by comparing:
- **Wave Summary Data**: Shows shipments with SKUs and Shipment IDs but without actual delivery dates
- **ColdCart Shipment Stats**: Confirms if orders were actually shipped and their status (delivered, in transit, etc.)
- **Shopify Orders**: Original order details for cross-reference

Use this page to verify how planned shipments were fulfilled.
""")

def get_default_dates():
    """Get last week's Monday to Sunday"""
    today = datetime.now()
    last_week = today - timedelta(weeks=1)
    monday = last_week - timedelta(days=last_week.weekday())
    sunday = monday + timedelta(days=6)
    return monday.date(), sunday.date()

def get_coldcart_data(start_date, end_date):
    """Get joined ColdCart data"""
    creds = get_credentials()
    if hasattr(creds, 'with_scopes'):
        creds = creds.with_scopes(SCOPES)
    
    client = bigquery.Client(
        credentials=creds, 
        project='nca-toolkit-project-446011'
    )

    # Wave Summaries Schema (cc_wave_summaries):
    # - ShipmentID
    # - TrackingCode
    # - ColdcartID
    # - BoxName
    # - Sku
    # - Description
    # - Quantity
    # - WarehouseLocations
    # - BatchCodes
    # - LabelUrl
    # - Carrier
    # - DestinationName
    # - csv_filename
    
    # Shipment Stats Schema (cc_shipment_stats):
    # - ShipmentId
    # - ClientId
    # - OrderId
    # - ExternalOrderId
    # - OrderNumber
    # - OrderTypeId
    # - Tags
    # - StatusId
    # - StatusName
    # - CreatedDate
    # - ShippedDate
    # - DeliveredDate
    # - EstimatedWeightLb
    # - ShippingBoxId
    # - LengthIn
    # - DepthIn
    # - WidthIn
    # - TrackingCode
    # - LabelUrl
    # - ExternalCarrierAccountId
    # - OriginCity
    # - OriginState
    # - OriginPostalCode
    # - DestinationName
    # - DestinationStreet
    # - DestinationCity
    # - DestinationState
    # - DestinationPostalCode
    # - CarrierName
    # - ServiceName
    # - FulfillmentWarehouseId
    # - DaysInTransit
    # - EstDaysInTransit
    # - HasAnomaly
    # - IsBatched
    # - FulfillmentCost
    # - ServiceFee
    # - UnitOverageCost
    # - LastMileCost
    # - CoolantCost
    # - InnerPackagingCost
    # - MarketingInsertCost
    # - BoxCost
    # - Total
    
    query = f"""
    WITH filtered_waves AS (
        SELECT *
        FROM `nca-toolkit-project-446011.fulfillment_cc_shopify.cc_wave_summaries`
    )
    SELECT w.*, s.*
    FROM filtered_waves w
    LEFT JOIN `nca-toolkit-project-446011.fulfillment_cc_shopify.cc_shipment_stats` s
        ON w.ShipmentID = s.ShipmentId
    WHERE DATE(PARSE_TIMESTAMP('%m/%d/%Y %H:%M:%S', s.CreatedDate)) 
        BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY PARSE_TIMESTAMP('%m/%d/%Y %H:%M:%S', s.CreatedDate) DESC
    """
    
    return client.query(query).to_dataframe()

def get_shopify_data(start_date, end_date):
    """Get Shopify fulfilled orders"""
    return update_fulfilled_orders_data(
        start_date=datetime.combine(start_date, datetime.min.time()),
        end_date=datetime.combine(end_date, datetime.max.time())
    )

# Get default dates
default_start, default_end = get_default_dates()

# Date range selector
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=default_end,
        max_value=datetime.now(),
        min_value=start_date
    )

# Load data first
with st.spinner("Loading data..."):
    cc_df = get_coldcart_data(start_date, end_date)
    try:
        shopify_df = get_shopify_data(start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching Shopify data: {str(e)}")
        shopify_df = pd.DataFrame()

# Global search for all tables
search_term = st.text_input("🔍 Search across all data", help="Search will be applied to all tables below")

# Function to filter dataframe based on search term
def filter_dataframe(df, search):
    if search:
        mask = df.astype(str).apply(
            lambda x: x.str.contains(search, case=False, na=False)
        ).any(axis=1)
        return df[mask]
    return df

# ColdCart Data section
st.subheader("ColdCart Data")
st.caption("ColdCart Wave Summaries and Orders Shipment Stats joined on ShipmentID, showing detailed fulfillment information")

# Calculate matching statistics
if not cc_df.empty:
    print("Available columns:", cc_df.columns.tolist())  # Debug print to see actual columns
    
    # Count total unique wave ShipmentIDs
    total_waves = len(cc_df['ShipmentID'].unique())  # From wave_summaries (w.*)
    # Count waves that have matching shipment stats
    matching_waves = len(cc_df[cc_df['ShipmentId_1'].notna()]['ShipmentID'].unique())  # ShipmentId_1 from shipment_stats (s.*)
    match_rate = round((matching_waves / total_waves * 100), 1) if total_waves > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Wave Summaries", total_waves)
    with col2:
        st.metric("With Shipment Stats", matching_waves)
    with col3:
        st.metric("Match Rate", f"{match_rate}%")

# Separate Wave Summaries Table
st.subheader("🌊 Wave Summaries")
st.caption("Wave summaries data from cc_wave_summaries table")
if not cc_df.empty:
    # Get wave summaries columns (those from the w.* part of the join)
    wave_columns = [
        'ShipmentID', 'TrackingCode', 'ColdcartID', 'BoxName', 'Sku', 
        'Description', 'Quantity', 'WarehouseLocations', 'BatchCodes', 
        'LabelUrl', 'Carrier', 'DestinationName', 'csv_filename'
    ]
    
    # Filter to only existing columns
    available_wave_columns = [col for col in wave_columns if col in cc_df.columns]
    wave_only_df = cc_df[available_wave_columns].drop_duplicates()
    
    # Apply search filter
    filtered_wave_df = filter_dataframe(wave_only_df, search_term)
    
    st.text(f"Showing {len(filtered_wave_df)} wave summary rows")
    st.dataframe(filtered_wave_df, use_container_width=True)
else:
    st.warning("No wave summaries data available.")

st.markdown("---")

# Separate Shipment Stats Table
st.subheader("📊 Shipment Stats")
st.caption("Shipment stats data from cc_shipment_stats table (delivery tracking and costs)")
if not cc_df.empty:
    # Get rows that have shipment stats data (non-null ShipmentId_1)
    stats_df = cc_df[cc_df['ShipmentId_1'].notna()].copy()
    
    if not stats_df.empty:
        # Get shipment stats columns (those from the s.* part of the join)
        # Note: BigQuery adds _1 suffix to duplicate column names from the second table
        stats_columns = [
            'ShipmentId_1', 'ClientId', 'OrderId', 'ExternalOrderId', 'OrderNumber',
            'OrderTypeId', 'Tags', 'StatusId', 'StatusName', 'CreatedDate',
            'ShippedDate', 'DeliveredDate', 'EstimatedWeightLb', 'ShippingBoxId',
            'LengthIn', 'DepthIn', 'WidthIn', 'TrackingCode_1', 'LabelUrl_1',
            'ExternalCarrierAccountId', 'OriginCity', 'OriginState', 'OriginPostalCode',
            'DestinationName_1', 'DestinationStreet', 'DestinationCity', 'DestinationState',
            'DestinationPostalCode', 'CarrierName', 'ServiceName', 'FulfillmentWarehouseId',
            'DaysInTransit', 'EstDaysInTransit', 'HasAnomaly', 'IsBatched',
            'FulfillmentCost', 'ServiceFee', 'UnitOverageCost', 'LastMileCost',
            'CoolantCost', 'InnerPackagingCost', 'MarketingInsertCost', 'BoxCost', 'Total'
        ]
        
        # Filter to only existing columns
        available_stats_columns = [col for col in stats_columns if col in stats_df.columns]
        stats_only_df = stats_df[available_stats_columns].drop_duplicates()
        
        # Apply search filter
        filtered_stats_df = filter_dataframe(stats_only_df, search_term)
        
        st.text(f"Showing {len(filtered_stats_df)} shipment stats rows")
        st.dataframe(filtered_stats_df, use_container_width=True)
    else:
        st.warning("No shipment stats data available for the selected date range.")
else:
    st.warning("No shipment stats data available.")

st.markdown("---")

# Create comprehensive join with Shopify orders
st.subheader("🔗 Complete Order Analysis")
st.caption("ColdCart data joined with Shopify orders by SKU and Order ID")

if not cc_df.empty and not shopify_df.empty:
    # Initialize DataProcessor for SKU mapping
    data_processor = DataProcessor()
    data_processor.load_sku_mappings()
    
    # First join by order ID to get fulfillment center info
    cc_for_join = cc_df.copy()
    shopify_for_join = shopify_df.copy()
    
    # Clean order IDs for joining
    cc_for_join['ExternalOrderId_clean'] = cc_for_join['ExternalOrderId'].astype(str).str.strip()
    shopify_for_join['Order Name_clean'] = shopify_for_join['Order Name'].astype(str).str.strip()
    
    # Join on order ID
    merged_df = pd.merge(
        cc_for_join,
        shopify_for_join,
        left_on='ExternalOrderId_clean',
        right_on='Order Name_clean',
        how='outer',
        suffixes=('_cc', '_shopify')
    )
    
    # Add match type column
    merged_df['match_type'] = 'No Match'
    
    # For each row, check if Shopify SKU matches inventory SKU using proper mapping
    for idx, row in merged_df.iterrows():
        cc_sku = row['Sku'] if pd.notna(row['Sku']) else None
        shopify_sku = row['SKU'] if pd.notna(row['SKU']) else None
        fc = row.get('OriginCity', 'Oxnard')  # Default to Oxnard if not specified
        
        if cc_sku and shopify_sku:
            # Get inventory SKU for this Shopify SKU
            mapped_sku = data_processor.map_shopify_to_inventory_sku(
                shopify_sku=shopify_sku,
                fulfillment_center=fc
            )
            
            if mapped_sku and mapped_sku.lower() == cc_sku.lower():
                merged_df.at[idx, 'match_type'] = 'SKU Match'
            elif pd.notna(row['ExternalOrderId']) and pd.notna(row['Order Name']):
                merged_df.at[idx, 'match_type'] = 'Order ID Match Only'
    
    # Calculate statistics
    stats = {
        'total_cc_orders': len(cc_df['ExternalOrderId'].dropna().unique()),
        'total_shopify_orders': len(shopify_df['Order Name'].dropna().unique()),
        'sku_matches': len(merged_df[merged_df['match_type'] == 'SKU Match']),
        'order_matches': len(merged_df[merged_df['match_type'] == 'Order ID Match Only']),
        'unmatched': len(merged_df[merged_df['match_type'] == 'No Match'])
    }
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SKU Matches", stats['sku_matches'])
        st.metric("Total ColdCart Orders", stats['total_cc_orders'])
    
    with col2:
        st.metric("Order ID Only Matches", stats['order_matches'])
        st.metric("Total Shopify Orders", stats['total_shopify_orders'])
    
    with col3:
        st.metric("Unmatched Records", stats['unmatched'])
        match_rate = round((stats['sku_matches'] / max(stats['total_cc_orders'], stats['total_shopify_orders']) * 100), 1)
        st.metric("Match Rate", f"{match_rate}%")
    
    # Show match type distribution
    match_type_counts = merged_df['match_type'].value_counts()
    st.write("**Match Type Distribution:**")
    st.write(match_type_counts)
    
    # Apply search filter
    filtered_df = filter_dataframe(merged_df, search_term)
    
    st.text(f"Showing {len(filtered_df)} rows")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Add download buttons
    st.markdown("---")
    st.subheader("Download Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not cc_df.empty:
            csv = cc_df.to_csv(index=False)
            st.download_button(
                "Download ColdCart Data",
                csv,
                "coldcart_data.csv",
                "text/csv",
                key='download-cc'
            )
    
    with col2:
        if not shopify_df.empty:
            csv = shopify_df.to_csv(index=False)
            st.download_button(
                "Download Shopify Data",
                csv,
                "shopify_orders.csv",
                "text/csv",
                key='download-shopify'
            )
    
    with col3:
        if not filtered_df.empty:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download Merged Data",
                csv,
                "merged_data.csv",
                "text/csv",
                key='download-merged'
            )
else:
    st.warning("Need both ColdCart and Shopify data to create analysis.")
