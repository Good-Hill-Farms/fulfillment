import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from utils.google_sheets import get_credentials
from utils.scripts_shopify.shopify_orders_report import update_fulfilled_orders_data

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
    
    query = f"""
    SELECT *
    FROM `nca-toolkit-project-446011.fulfillment_cc_shopify.cc_shipment_stats` s
    LEFT JOIN `nca-toolkit-project-446011.fulfillment_cc_shopify.cc_wave_summaries` w
        ON s.ShipmentId = w.ShipmentID
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
# Apply global search to ColdCart data
filtered_cc_df = filter_dataframe(cc_df, search_term)

# Calculate matching statistics
if not cc_df.empty:
    total_shipments = len(cc_df['ShipmentId'].unique())
    matching_shipments = len(cc_df[cc_df['ShipmentID_1'].notna()]['ShipmentId'].unique())
    match_rate = round((matching_shipments / total_shipments * 100), 1) if total_shipments > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Shipments", total_shipments)
    with col2:
        st.metric("Matched with Wave", matching_shipments)
    with col3:
        st.metric("Match Rate", f"{match_rate}%")

# Show ColdCart row count and data
st.text(f"Showing {len(filtered_cc_df)} ColdCart rows")
st.dataframe(filtered_cc_df, use_container_width=True)

# Add separator
st.markdown("---")

# Shopify Data section
st.subheader("Shopify Fulfilled Orders")
st.caption("Completed orders from Shopify with order details, products, and delivery information")
if not shopify_df.empty:
    # Apply global search to Shopify data
    filtered_shopify_df = filter_dataframe(shopify_df, search_term)
    
    # Show Shopify row count and data
    st.text(f"Showing {len(filtered_shopify_df)} Shopify orders")
    st.dataframe(filtered_shopify_df, use_container_width=True)
else:
    st.warning("No Shopify data available.")

# Add download buttons
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if not filtered_cc_df.empty:
        csv = cc_df.to_csv(index=False)
        st.download_button(
            "Download ColdCart Data",
            csv,
            "coldcart_data.csv",
            "text/csv",
            key='download-cc'
        )

with col2:
    if not filtered_shopify_df.empty:
        csv = shopify_df.to_csv(index=False)
        st.download_button(
            "Download Shopify Data",
            csv,
            "shopify_orders.csv",
            "text/csv",
            key='download-shopify'
        )
