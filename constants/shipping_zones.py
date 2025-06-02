"""
Shipping zones constants and utilities for fulfillment center selection.

This module defines the shipping zones data structure and provides utility functions
for working with shipping zones. Shipping zones are used to optimize fulfillment center
selection based on customer ZIP codes.
"""
import os

import pandas as pd

# Fulfillment center information
FULFILLMENT_CENTERS = {
    "moorpark": {
        "name": "Moorpark",
        "state": "CA",
        "zip": "93021",
        "default_zone": None,  # No default zone - must be determined by ZIP code
    },
    "wheeling": {
        "name": "Wheeling",
        "state": "IL",
        "zip": "60090",
        "default_zone": None,  # No default zone - must be determined by ZIP code
    },
}

# Default column names for shipping zones data
SHIPPING_ZONE_COLUMNS = {
    "zip_prefix": "zip_prefix",
    "moorpark_zone": "moorpark_zone",
    "wheeling_zone": "wheeling_zone",
}

# Default path for shipping zones file
DEFAULT_SHIPPING_ZONES_PATH = os.path.join("constants", "shipping_zones.csv")


def load_shipping_zones(file_path=None):
    """
    Load shipping zones data from a CSV file.

    Args:
        file_path: Path to the shipping zones CSV file or file object

    Returns:
        pandas.DataFrame: Processed shipping zones dataframe with zip_prefix, moorpark_zone, and wheeling_zone columns
    """
    try:
        # If file_path is provided, try to load from it
        if file_path is not None:
            if isinstance(file_path, str):
                # It's a file path
                raw_data = pd.read_csv(file_path, header=None)
            else:
                # It's a file object
                raw_data = pd.read_csv(file_path, header=None)
        else:
            # Try to load from default location
            raw_data = pd.read_csv(DEFAULT_SHIPPING_ZONES_PATH, header=None)

        # Check if the file has the expected format (based on the first few rows)
        if raw_data.shape[1] >= 5:  # We expect at least 5 columns in the side-by-side format
            # Skip the header rows (first 2 rows)
            data = raw_data.iloc[2:].copy()

            # Extract the ZIP code prefixes and zones
            result_data = []

            for _, row in data.iterrows():
                zip_prefix = row[0]  # Moorpark ZIP prefix
                moorpark_zone = row[1]  # Moorpark zone
                wheeling_zip = row[3]  # Wheeling ZIP prefix (should be the same as Moorpark)
                wheeling_zone = row[4]  # Wheeling zone

                # Ensure ZIP prefixes are strings and zones are integers
                try:
                    zip_prefix = str(int(zip_prefix)).zfill(3)  # Ensure 3-digit format
                    moorpark_zone = int(moorpark_zone) if pd.notna(moorpark_zone) else None
                    wheeling_zone = int(wheeling_zone) if pd.notna(wheeling_zone) else None

                    result_data.append(
                        {
                            "zip_prefix": zip_prefix,
                            "moorpark_zone": moorpark_zone,
                            "wheeling_zone": wheeling_zone,
                        }
                    )
                except (ValueError, TypeError):
                    # Skip rows with invalid data
                    print(f"Skipping row with invalid data: {row.values}")

            # Create a new DataFrame with the extracted data
            shipping_zones_df = pd.DataFrame(result_data)

            # Ensure all required columns exist
            required_columns = ["zip_prefix", "moorpark_zone", "wheeling_zone"]
            for col in required_columns:
                if col not in shipping_zones_df.columns:
                    shipping_zones_df[col] = None

            return shipping_zones_df
        else:
            # If the file doesn't have the expected format, try to load it as a standard CSV
            print(
                "CSV format doesn't match expected side-by-side format. Trying standard format..."
            )
            shipping_zones_df = pd.read_csv(
                file_path if file_path is not None else DEFAULT_SHIPPING_ZONES_PATH
            )

            # Ensure the dataframe has the required columns
            required_columns = ["zip_prefix", "moorpark_zone", "wheeling_zone"]

            # Check if all required columns exist
            if not all(col in shipping_zones_df.columns for col in required_columns):
                # Try to rename columns if they exist with different names
                column_mapping = {}

                # Common column name variations
                mappings = {
                    "zip_prefix": ["zip_prefix", "zip", "zip code", "zipcode", "ZIP Code Prefix"],
                    "moorpark_zone": [
                        "moorpark_zone",
                        "moorpark zone",
                        "moorpark",
                        "Moorpark Zone",
                        "Zone",
                    ],
                    "wheeling_zone": [
                        "wheeling_zone",
                        "wheeling zone",
                        "wheeling",
                        "Wheeling Zone",
                    ],
                }

                for target_col, candidates in mappings.items():
                    for candidate in candidates:
                        if candidate in shipping_zones_df.columns:
                            column_mapping[candidate] = target_col
                            break

                # Rename columns if mapping exists
                if column_mapping:
                    shipping_zones_df = shipping_zones_df.rename(columns=column_mapping)

            # If we still don't have all required columns, raise an error
            if not all(col in shipping_zones_df.columns for col in required_columns):
                raise ValueError(
                    f"Shipping zones CSV must contain the following columns: {required_columns}"
                )

            return shipping_zones_df

    except Exception as e:
        print(f"Error loading shipping zones: {e}")
        # Return an empty DataFrame with the required columns
        return pd.DataFrame(columns=["zip_prefix", "moorpark_zone", "wheeling_zone"])


def get_zone_by_zip(shipping_zones_df, zip_code, fulfillment_center):
    """
    Get the shipping zone for a specific ZIP code and fulfillment center.

    Args:
        shipping_zones_df: DataFrame containing shipping zone data
        zip_code: Customer ZIP code
        fulfillment_center: Name of the fulfillment center ('moorpark' or 'wheeling')

    Returns:
        int or None: Shipping zone number or None if not found
    """
    if not zip_code or len(zip_code) < 3 or shipping_zones_df.empty:
        return None

    # Get the first 3 digits of the ZIP code for zone lookup
    zip_prefix = zip_code[:3]

    # Get the zone column name for the specified fulfillment center
    zone_column = f"{fulfillment_center}_zone"

    # Check if the necessary columns exist
    if (
        "zip_prefix" not in shipping_zones_df.columns
        or zone_column not in shipping_zones_df.columns
    ):
        return None

    # Find the matching ZIP prefix
    zip_match = shipping_zones_df[shipping_zones_df["zip_prefix"] == zip_prefix]

    if not zip_match.empty:
        zone_value = zip_match[zone_column].iloc[0]
        if pd.notna(zone_value):
            try:
                return int(zone_value)
            except (ValueError, TypeError):
                print(f"Invalid {fulfillment_center} zone value: {zone_value}")

    return None


def estimate_zone_by_geography(zip_code, fulfillment_center):
    """
    Estimate shipping zone based on geographic heuristics when zone data is not available.

    Args:
        zip_code: Customer ZIP code
        fulfillment_center: Name of the fulfillment center ('moorpark' or 'wheeling')

    Returns:
        int: Estimated shipping zone (1-8)
    """
    if not zip_code or len(zip_code) < 1:
        return 4  # Default to middle zone if no ZIP code

    first_digit = int(zip_code[0])

    if fulfillment_center == "moorpark":
        # For Moorpark (CA): West Coast (closer) = lower zone, East Coast (further) = higher zone
        if first_digit >= 9:  # West Coast
            return 1
        elif first_digit >= 8:  # Mountain West
            return 2
        elif first_digit >= 7:  # Southwest
            return 3
        elif first_digit >= 6:  # Midwest
            return 5
        else:  # East Coast
            return 7
    else:  # wheeling
        # For Wheeling (IL): East Coast/Midwest (closer) = lower zone, West Coast (further) = higher zone
        if first_digit <= 2:  # Northeast
            return 2
        elif first_digit <= 3:  # Mid-Atlantic
            return 1
        elif first_digit <= 5:  # Southeast/Midwest
            return 2
        elif first_digit <= 6:  # South/Central
            return 3
        elif first_digit <= 7:  # Southwest
            return 5
        else:  # West Coast
            return 7
