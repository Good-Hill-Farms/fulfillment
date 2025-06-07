import json
import logging
import os
import pandas as pd

# do not change
orders_columns = [
    "Date",
    "Name",
    "order id",
    "Customer: First Name",
    "Customer: Last Name",
    "Email",
    "Shipping: Name",
    "Shipping: Address 1",
    "Shipping: Address 2",
    "Shipping: City",
    "Shipping: Province Code",
    "Shipping: Zip",
    "Note",
    "SKU Helper",
    "Line: Fulfillable Quantity",
    "Line: ID",
    "NEW Tags",
    "MAX PKG NUM",
    "Fulfillment Center",
    "Saturday Shipping"
]

# do not change
inventory_columns = [
    "WarehouseName",
    "ItemId",
    "Sku",
    "Name",
    "Type",
    "BatchCode",
    "AvailableQty",
    "DaysOnHand",
    "Balance",
]

logger = logging.getLogger(__name__)

class DataParser:
    def __init__(self):
        """Initialize the DataParser with required tracking variables."""
        self._logged_issues = set()

    def parse_orders(self, orders_file) -> pd.DataFrame:
        """Load and preprocess orders CSV file.

        Args:
            orders_file: Path to the orders CSV file

        Returns:
            pandas.DataFrame: Processed orders dataframe
        """
        try:
            # Read CSV file with all columns
            df = pd.read_csv(orders_file, na_values=[''], keep_default_na=True)
            
            # Identify column types
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            formula_cols = [col for col in df.columns if isinstance(col, str) and 
                          ('isnumber' in col.lower() or 'search' in col.lower())]
            tracking_cols = [col for col in df.columns if isinstance(col, str) and col.isdigit()]
            
            # Drop unnecessary columns
            cols_to_drop = unnamed_cols + formula_cols + tracking_cols
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} unnecessary columns")

            # Normalize column names
            df.columns = [str(col).strip() for col in df.columns]

            # Handle SKU columns
            if 'SKU Helper' in df.columns:
                df['SKU'] = df['SKU Helper'].fillna('')
            elif 'SKU' not in df.columns:
                df['SKU'] = 'unknown'

            # Clean up shipping info
            if 'Shipping: Zip' in df.columns:
                df['Shipping: Zip'] = df['Shipping: Zip'].astype(str).str.strip()
                # Extract just the ZIP code (first 5-9 digits)
                df['Shipping: Zip'] = df['Shipping: Zip'].str.extract(r'(\d{5}(?:-?\d{4})?)', expand=False).fillna('')
            else:
                df['Shipping: Zip'] = ''

            # Clean up other address fields
            address_cols = ['Shipping: Address 1', 'Shipping: Address 2', 'Shipping: City', 'Shipping: Province Code']
            for col in address_cols:
                if col in df.columns:
                    # Remove multiple spaces and normalize whitespace
                    df[col] = df[col].fillna('').astype(str).str.strip()
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            # Ensure all required columns exist
            for col in orders_columns:
                if col not in df.columns:
                    logger.warning(f"Required column '{col}' not found in orders file")
                    df[col] = None

            # Basic preprocessing
            # Convert date format
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            else:
                df["Date"] = pd.to_datetime("today")

            # Fill NA values
            for col in ["Shipping: Address 2", "Note", "NEW Tags"]:
                df[col] = df[col].fillna("")

            # Handle numeric columns
            df["MAX PKG NUM"] = pd.to_numeric(df["MAX PKG NUM"], errors="coerce").fillna(1)
            df["Line: Fulfillable Quantity"] = pd.to_numeric(df["Line: Fulfillable Quantity"], errors="coerce").fillna(1)

            # Clean up SKU format
            if "SKU Helper" in df.columns:
                # Remove 'f.' prefix and any trailing commas or whitespace
                df["SKU"] = df["SKU Helper"].str.replace("f.", "", regex=False)
                df["SKU"] = df["SKU"].str.replace(",", "", regex=False)
                df["SKU"] = df["SKU"].str.strip()
            elif "SKU" not in df.columns:
                df["SKU"] = "unknown"
                
            # Drop any extra columns that aren't in orders_columns
            extra_cols = [col for col in df.columns if col not in orders_columns + ['SKU']]
            if extra_cols:
                df = df.drop(columns=extra_cols)
                logger.info(f"Dropped extra columns: {extra_cols}")
            
            # Parse zip codes
            df["Shipping: Zip"] = df["Shipping: Zip"].astype(str).str.strip().str.split("-").str[0]

            return df

        except Exception as e:
            logger.error(f"Error processing orders file: {str(e)}")
            return pd.DataFrame(columns=orders_columns)

    def parse_inventory(self, inventory_file) -> pd.DataFrame:
        """Load and preprocess inventory CSV file from the main data section.

        Args:
            inventory_file: Path to the inventory CSV file

        Returns:
            pandas.DataFrame: Raw inventory dataframe with basic cleaning and proper grouping
        """
        try:
            # Read CSV file, using first row as column names
            df = pd.read_csv(inventory_file, header=0)
            
            # Rename unnamed columns to their actual names from first row
            unnamed_cols = {col: df.iloc[0][col] for col in df.columns if col.startswith('Unnamed:')}
            df = df.rename(columns=unnamed_cols)
            
            # Skip the first row since it was column names
            df = df.iloc[1:].copy()
            
            # Select only the columns we need, adding any missing columns with default values
            for col in inventory_columns:
                if col not in df.columns:
                    logger.warning(f"Required column '{col}' not found in inventory file, adding with default values")
                    df[col] = None if col not in ['AvailableQty', 'Balance', 'DaysOnHand'] else 0
            
            # Convert numeric columns, handling commas
            for col in ['AvailableQty', 'Balance']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Ensure we're tracking by SKU and warehouse properly
            # Each row is kept as is - no summing of Balance values
            # This creates a composite key for tracking but doesn't modify the data
            df['tracking_key'] = df['WarehouseName'] + '|' + df['Sku']
            logger.info(f"Parsed inventory with {len(df)} rows, {df['tracking_key'].nunique()} unique SKU-warehouse combinations")

            return df

        except Exception as e:
            logger.error(f"Error processing inventory file: {str(e)}")
            return pd.DataFrame(columns=[
                'WarehouseName', 'Sku', 'Name', 'Type', 'Balance', 'Status'
            ])

if __name__ == "__main__":
    parser = DataParser()
    orders = parser.parse_orders("docs/orders.csv")
    inventory = parser.parse_inventory("docs/inventory.csv")
    with open("parsed_orders.csv", "w") as f:
        orders.to_csv(f, index=False)
    with open("parsed_inventory.csv", "w") as f:
        inventory.to_csv(f, index=False)
    
