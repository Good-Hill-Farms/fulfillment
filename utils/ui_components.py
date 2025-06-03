from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


def render_header():
    """Render the application header"""
    st.title("Mixy Matchi Fulfillment Assistant")
    st.markdown(
        """
    This application helps assign customer fruit orders to fulfillment centers using:
    - Rule-based assignments
    - Inventory optimization
    - Shortage detection and substitution suggestions
    """
    )
    
    # Display inventory shortage summary if available
    if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
        with st.expander(f"⚠️ INVENTORY SHORTAGES DETECTED: {len(st.session_state.shortage_summary)} items", expanded=True):
            # Display grouped shortage summary with totals if available
            if "grouped_shortage_summary" in st.session_state and not st.session_state.grouped_shortage_summary.empty:
                st.markdown("### Grouped Shortage Summary with Totals")
                
                # Check if issue column exists in the grouped summary
                grouped_df = st.session_state.grouped_shortage_summary.copy()
                if 'issue' in grouped_df.columns:
                    # Highlight rows with issues
                    st.markdown("**⚠️ Rows with issues are highlighted**")
                    # Create a styled dataframe
                    styled_df = grouped_df.style.apply(
                        lambda row: ['background-color: #ffcccc' if row['issue'] else '' for _ in row],
                        axis=1
                    )
                    st.dataframe(styled_df)
                else:
                    st.dataframe(grouped_df)
                
                # Provide download button for grouped shortage summary
                grouped_csv = st.session_state.grouped_shortage_summary.to_csv(index=False)
                st.download_button(
                    label="Download Grouped Shortage Summary",
                    data=grouped_csv,
                    file_name="grouped_shortage_summary.csv",
                    mime="text/csv",
                    key="download_grouped_shortage"
                )
            
            # Display detailed shortage summary
            st.markdown("### Detailed Shortage Summary")
            
            # Check if issue column exists in the detailed summary
            detailed_df = st.session_state.shortage_summary.copy()
            
            # Also highlight rows with empty fulfillment center
            if 'fulfillment_center' in detailed_df.columns:
                # Create a styled dataframe that highlights rows with empty fulfillment center
                styled_df = detailed_df.style.apply(
                    lambda row: ['background-color: #ffcccc' if ('issue' in row.index and row['issue']) or 
                                 ('fulfillment_center' in row.index and not row['fulfillment_center']) 
                                 else '' for _ in row],
                    axis=1
                )
                st.dataframe(styled_df)
            else:
                st.dataframe(detailed_df)
            
            # Provide download button for detailed shortage summary
            csv = st.session_state.shortage_summary.to_csv(index=False)
            st.download_button(
                label="Download Detailed Shortage Summary",
                data=csv,
                file_name="shortage_summary.csv",
                mime="text/csv",
                key="download_detailed_shortage"
            )
            
            st.markdown("### Substitution Suggestions")
            st.info("The system has automatically suggested possible substitutions for missing items based on similar weight and type. Check the 'Issues' column in the orders tab for detailed suggestions.")
    
    # Display inventory summary if available
    if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty:
        with st.expander("📊 Current Inventory Summary"):
            st.dataframe(st.session_state.inventory_summary)
            
            # Provide download button for inventory summary
            csv = st.session_state.inventory_summary.to_csv(index=False)
            st.download_button(
                label="Download Inventory Summary",
                data=csv,
                file_name="inventory_summary.csv",
                mime="text/csv",
            )


def render_progress_bar(current_step, total_steps, step_name):
    """Render a progress bar for processes"""
    progress = st.progress(0)
    # Avoid division by zero
    if total_steps > 0:
        progress.progress(current_step / total_steps)
    else:
        progress.progress(0)
    st.caption(f"Step {current_step}/{total_steps}: {step_name}")


def render_inventory_analysis(processed_orders, inventory_df):
    """
    Render inventory analysis showing current inventory and projected remaining inventory after orders
    
    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
    """
    if processed_orders is None or inventory_df is None:
        st.warning("No data available for inventory analysis")
        return
        
    st.subheader("📊 Inventory Analysis")
    
    # Create tabs for different inventory views
    inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Current Inventory", "Projected Remaining Inventory", "Inventory Changes"])
    
    # Process inventory data
    # Ensure inventory_df has the necessary columns
    if 'Sku' not in inventory_df.columns or 'WarehouseName' not in inventory_df.columns:
        with inv_tab1:
            st.error("Inventory data is missing required columns (Sku, WarehouseName)")
        return
    
    # Group inventory by SKU and warehouse
    inventory_summary = inventory_df.groupby(['WarehouseName', 'Sku']).agg({
        'AvailableQty': 'sum',
        'Balance': 'first'
    }).reset_index()
    
    # Calculate order quantities by SKU
    if 'sku' in processed_orders.columns and 'Transaction Quantity' in processed_orders.columns:
        order_quantities = processed_orders.groupby('sku').agg({
            'Transaction Quantity': 'sum',
            'Fulfillment Center': 'first'
        }).reset_index()
        
        # Create a copy of inventory for projected calculations
        projected_inventory = inventory_summary.copy()
        
        # Map fulfillment centers to warehouse names
        fc_to_warehouse = {
            'Moorpark': 'CA-Moorpark-93021',
            'CA-Moorpark-93021': 'CA-Moorpark-93021',
            'Oxnard': 'CA-Oxnard-93030',
            'CA-Oxnard-93030': 'CA-Oxnard-93030',
            'Wheeling': 'IL-Wheeling-60090',
            'IL-Wheeling-60090': 'IL-Wheeling-60090'
        }
        
        # Create a new column for projected remaining balance
        projected_inventory['Projected Remaining'] = projected_inventory['Balance']
        
        # Update projected remaining based on orders
        for _, order_row in order_quantities.iterrows():
            sku = order_row['sku']
            qty = order_row['Transaction Quantity']
            fc = order_row['Fulfillment Center']
            
            # Convert fulfillment center to warehouse name
            warehouse = fc_to_warehouse.get(fc, fc)
            
            # Find matching inventory row
            matching_rows = projected_inventory[
                (projected_inventory['Sku'] == sku) & 
                (projected_inventory['WarehouseName'] == warehouse)
            ]
            
            if not matching_rows.empty:
                idx = matching_rows.index[0]
                # Update projected remaining
                current = projected_inventory.loc[idx, 'Projected Remaining']
                projected_inventory.loc[idx, 'Projected Remaining'] = max(0, current - qty)
        
        # Calculate the change in inventory
        projected_inventory['Change'] = projected_inventory['Projected Remaining'] - projected_inventory['Balance']
        
        # Display current inventory
        with inv_tab1:
            st.write("Current inventory levels before processing orders:")
            st.dataframe(
                inventory_summary[['WarehouseName', 'Sku', 'AvailableQty', 'Balance']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'WarehouseName': st.column_config.TextColumn('Warehouse'),
                    'Sku': st.column_config.TextColumn('SKU'),
                    'AvailableQty': st.column_config.NumberColumn('Available Qty'),
                    'Balance': st.column_config.NumberColumn('Current Balance')
                }
            )
        
        # Display projected remaining inventory
        with inv_tab2:
            st.write("Projected inventory levels after processing all orders:")
            st.dataframe(
                projected_inventory[['WarehouseName', 'Sku', 'Balance', 'Projected Remaining']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'WarehouseName': st.column_config.TextColumn('Warehouse'),
                    'Sku': st.column_config.TextColumn('SKU'),
                    'Balance': st.column_config.NumberColumn('Current Balance'),
                    'Projected Remaining': st.column_config.NumberColumn('Projected Remaining')
                }
            )
        
        # Display inventory changes
        with inv_tab3:
            # Filter to only show items with changes
            changes_df = projected_inventory[projected_inventory['Change'] < 0].copy()
            changes_df['Change'] = changes_df['Change'].abs()  # Make positive for display
            
            if not changes_df.empty:
                st.write("Inventory items affected by orders:")
                st.dataframe(
                    changes_df[['WarehouseName', 'Sku', 'Balance', 'Projected Remaining', 'Change']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'WarehouseName': st.column_config.TextColumn('Warehouse'),
                        'Sku': st.column_config.TextColumn('SKU'),
                        'Balance': st.column_config.NumberColumn('Current Balance'),
                        'Projected Remaining': st.column_config.NumberColumn('Projected Remaining'),
                        'Change': st.column_config.NumberColumn('Quantity Used')
                    }
                )
                
                # Create a bar chart showing the top items with the most changes
                top_changes = changes_df.sort_values('Change', ascending=False).head(10)
                
                if not top_changes.empty:
                    fig = px.bar(
                        top_changes,
                        x='Sku',
                        y='Change',
                        title='Top 10 SKUs by Quantity Used',
                        color='WarehouseName',
                        labels={'Change': 'Quantity Used', 'Sku': 'SKU'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No inventory changes detected from orders.")
    else:
        with inv_tab1:
            st.dataframe(inventory_summary[['WarehouseName', 'Sku', 'AvailableQty', 'Balance']])
        with inv_tab2:
            st.warning("Cannot calculate projected inventory - order data is missing required columns.")
        with inv_tab3:
            st.warning("Cannot calculate inventory changes - order data is missing required columns.")

def render_summary_dashboard(processed_orders, inventory_df):
    """
    Render summary dashboard with charts and metrics

    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
    """
    if processed_orders is None or inventory_df is None:
        st.warning("No data available for dashboard")
        return
        
    # Create a copy of processed_orders to avoid modifying the original
    processed_orders = processed_orders.copy()
    
    # Ensure externalorderid and id columns are string type to prevent data type mismatch
    for id_col in ['externalorderid', 'id']:
        if id_col in processed_orders.columns:
            processed_orders[id_col] = processed_orders[id_col].astype(str)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_orders = len(processed_orders["ordernumber"].unique())
        st.metric("Unique Orders", total_orders)

    with col2:
        total_items = len(processed_orders)
        st.metric("Total Items", total_items)

    with col3:
        fulfillment_centers = processed_orders["Fulfillment Center"].nunique()
        st.metric("Fulfillment Centers", fulfillment_centers)

    with col4:
        issues = processed_orders[processed_orders["Issues"] != ""].shape[0]
        st.metric("Issues", issues, delta=None, delta_color="inverse")

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Show total number of line items processed
        total_line_items = processed_orders.shape[0]
        st.metric("Line Items", total_line_items)

    with col2:
        # Check if Priority column exists
        if "Priority" in processed_orders.columns:
            priority_orders = processed_orders[processed_orders["Priority"] != ""].shape[0]
        else:
            priority_orders = 0
        st.metric("Priority Orders", priority_orders)

    with col3:
        # Check if Bundle column exists
        if "Bundle" in processed_orders.columns:
            bundle_orders = processed_orders[processed_orders["Bundle"] != ""].shape[0]
        else:
            bundle_orders = 0
        st.metric("Bundle Orders", bundle_orders)

    with col4:
        if "Inventory Status" in processed_orders.columns:
            critical_inventory = processed_orders[
                processed_orders["Inventory Status"] == "Critical"
            ].shape[0]
            st.metric(
                "Critical Inventory Items", critical_inventory, delta=None, delta_color="inverse"
            )

    # Orders by fulfillment center
    st.subheader("Orders by Fulfillment Center")
    if "Fulfillment Center" in processed_orders.columns:
        fc_counts = processed_orders["Fulfillment Center"].value_counts().reset_index()
        fc_counts.columns = ["Fulfillment Center", "Count"]
    else:
        # Create empty dataframe if Fulfillment Center column doesn't exist
        fc_counts = pd.DataFrame({"Fulfillment Center": ["No Data"], "Count": [0]})

    fig = px.bar(
        fc_counts,
        x="Fulfillment Center",
        y="Count",
        title="Order Distribution by Fulfillment Center",
        color="Fulfillment Center",
        text="Count",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Priority distribution
    if "Priority" in processed_orders.columns and processed_orders["Priority"].any():
        st.subheader("Orders by Priority")
        priority_counts = (
            processed_orders[processed_orders["Priority"] != ""]["Priority"]
            .value_counts()
            .reset_index()
        )
        priority_counts.columns = ["Priority", "Count"]

        fig = px.pie(
            priority_counts,
            values="Count",
            names="Priority",
            title="Order Distribution by Priority",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Inventory utilization
    st.subheader("Inventory Utilization")

    # Calculate inventory usage
    inventory_usage = (
        processed_orders.groupby("sku")
        .agg({"Transaction Quantity": "sum", "Starting Balance": "first"})
        .reset_index()
    )
    # Avoid division by zero
    # Convert to numeric first to ensure round() works properly
    inventory_usage["Transaction Quantity"] = pd.to_numeric(
        inventory_usage["Transaction Quantity"], errors="coerce"
    )
    inventory_usage["Starting Balance"] = pd.to_numeric(
        inventory_usage["Starting Balance"], errors="coerce"
    )

    # Calculate usage percentage with safeguards
    inventory_usage["Usage Percentage"] = inventory_usage.apply(
        lambda row: min(
            100, round((row["Transaction Quantity"] / row["Starting Balance"]) * 100, 1)
        )
        if row["Starting Balance"] > 0
        else 0,
        axis=1,
    )

    # Filter out rows with zero starting balance
    inventory_usage = inventory_usage[inventory_usage["Starting Balance"] > 0]

    # Sort by usage percentage
    inventory_usage = inventory_usage.sort_values("Usage Percentage", ascending=False).head(10)

    fig = px.bar(
        inventory_usage,
        x="sku",
        y="Usage Percentage",
        title="Top 10 SKUs by Inventory Usage (%)",
        color="Usage Percentage",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Issues summary
    st.subheader("Issues Summary")

    if processed_orders[processed_orders["Issues"] != ""].shape[0] > 0:
        issues_df = processed_orders[processed_orders["Issues"] != ""]
        # Display only columns that exist in the dataframe
        display_columns = ["ordernumber", "sku", "Fulfillment Center", "Issues"]
        # Check if 'Priority' column exists before adding it
        if "Priority" in issues_df.columns:
            display_columns.append("Priority")
        st.dataframe(issues_df[display_columns])
    else:
        st.info("No issues found in processed orders.")

    # Bundle analysis
    if "Bundle" in processed_orders.columns and processed_orders["Bundle"].any():
        st.subheader("Bundle Analysis")
        bundle_counts = (
            processed_orders[processed_orders["Bundle"] != ""]["Bundle"]
            .value_counts()
            .reset_index()
        )
        bundle_counts.columns = ["Bundle", "Count"]

        fig = px.bar(
            bundle_counts,
            x="Bundle",
            y="Count",
            title="Orders by Bundle Type",
            color="Bundle",
            text="Count",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Fulfillment Center Distribution
    st.subheader("Fulfillment Center Distribution")
    if "Fulfillment Center" in processed_orders.columns:
        # Create a simple count of orders by fulfillment center
        fc_counts = processed_orders["Fulfillment Center"].value_counts().reset_index()
        fc_counts.columns = ["Fulfillment Center", "Count"]

        # Calculate percentages
        total_orders = fc_counts["Count"].sum()
        if total_orders > 0:  # Avoid division by zero
            fc_counts["Percentage"] = (fc_counts["Count"] / total_orders * 100).round(1)

            fig = px.bar(
                fc_counts,
                x="Fulfillment Center",
                y="Percentage",
                title="Order Distribution by Fulfillment Center (%)",
                color="Fulfillment Center",
                text="Percentage",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No orders data available.")
    else:
        st.info("Fulfillment Center data not available in the processed orders.")

    # Issues summary
    st.subheader("Issues Summary")

    # Extract issues
    issues_df = processed_orders[processed_orders["Issues"] != ""].copy()

    if not issues_df.empty:
        # Count issue types
        issues_df["Issue Type"] = issues_df["Issues"].str.extract(r"([^;]+);")
        issue_counts = issues_df["Issue Type"].value_counts().reset_index()
        issue_counts.columns = ["Issue Type", "Count"]

        fig = px.pie(
            issue_counts, values="Count", names="Issue Type", title="Distribution of Issues"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show issues table
        with st.expander("View All Issues"):
            # Define column config for issues table
            issues_column_config = {
                "externalorderid": st.column_config.TextColumn(
                    "Order ID", help="External order identifier", width="medium"
                ),
                "sku": st.column_config.TextColumn(
                    "SKU", help="Stock keeping unit", width="medium"
                ),
                "Fulfillment Center": st.column_config.TextColumn(
                    "Fulfillment Center", help="Assigned fulfillment center", width="medium"
                ),
                "Issues": st.column_config.TextColumn(
                    "Issues", help="Identified issues with the order", width="large"
                ),
            }

            # Ensure externalorderid is string type to prevent data type mismatch
            issues_df_display = issues_df.copy()
            if "externalorderid" in issues_df_display.columns:
                issues_df_display["externalorderid"] = issues_df_display["externalorderid"].astype(str)
            
            # Using Streamlit's built-in filtering capability
            st.data_editor(
                issues_df_display[["externalorderid", "sku", "Fulfillment Center", "Issues"]],
                use_container_width=True,
                column_config=issues_column_config,
                hide_index=True,
                key="issues_table_editor",
            )
    else:
        st.success("No issues found!")


def render_chat_widget(processed_orders=None):
    """
    Render a chat widget for interacting with the fulfillment assistant

    Args:
        processed_orders: DataFrame containing processed orders data
    """
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Apply CSS to create the chat interface styling
    st.markdown(
        """
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a container for the chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Process new messages if available in session state
    if "new_prompt" in st.session_state and st.session_state.new_prompt:
        prompt = st.session_state.new_prompt
        st.session_state.new_prompt = ""

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        response = generate_fulfillment_assistant_response(prompt, processed_orders)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Force a rerun to update the chat display
        st.rerun()


def generate_fulfillment_assistant_response(prompt, processed_orders=None):
    """
    Generate a response from the fulfillment assistant based on user prompt

    Args:
        prompt: User's prompt text
        processed_orders: DataFrame containing processed orders data

    Returns:
        str: Assistant's response
    """
    # Process the user's query
    prompt_lower = prompt.lower()

    # Check if we have processed orders data
    if processed_orders is None or processed_orders.empty:
        return "I don't have any order data to analyze yet. Please upload and process order files first."

    # Handle delivery service related queries
    if any(
        keyword in prompt_lower
        for keyword in ["delivery", "shipping", "carrier", "service", "ups", "fedex"]
    ):
        # Check if we have delivery days information
        if (
            "delivery_days" in processed_orders.columns
            and "carrier_name" in processed_orders.columns
            and "service_name" in processed_orders.columns
        ):
            # Create a summary of delivery services with days
            response = "Here's a breakdown of the delivery services used:\n\n"

            # Group by carrier, service, and days
            service_summary = (
                processed_orders.groupby(["carrier_name", "service_name", "delivery_days"])
                .size()
                .reset_index(name="count")
            )

            # Format the response by carrier
            current_carrier = None
            for _, row in service_summary.sort_values(["carrier_name", "service_name"]).iterrows():
                carrier = row["carrier_name"]
                service_name = row["service_name"]
                days = row["delivery_days"]
                count = row["count"]

                if current_carrier != carrier:
                    response += f"\n**{carrier}**:\n"
                    current_carrier = carrier

                response += f"- {service_name}: {count} orders ({days} days delivery time)\n"

            # Add average delivery time information
            avg_days = processed_orders["delivery_days"].mean()
            response += f"\nAverage delivery time across all orders: {avg_days:.1f} days\n"

            # Add information about fastest and slowest services
            if not service_summary.empty:
                fastest = service_summary.loc[service_summary["delivery_days"].idxmin()]
                slowest = service_summary.loc[service_summary["delivery_days"].idxmax()]

                response += f"\nFastest service: {fastest['carrier_name']} {fastest['service_name']} ({fastest['delivery_days']} days)\n"
                response += f"Slowest service: {slowest['carrier_name']} {slowest['service_name']} ({slowest['delivery_days']} days)\n"

        # Fall back to the old method if we don't have the new columns
        elif "preferredcarrierserviceid" in processed_orders.columns:
            delivery_counts = processed_orders["preferredcarrierserviceid"].value_counts().to_dict()
            response = "Here's a breakdown of the delivery services used:\n\n"

            # Group by carrier and service
            carrier_services = {}
            for service, count in delivery_counts.items():
                if pd.notna(service) and service:
                    carrier, service_name = service.split("_") if "_" in service else (service, "")
                    if carrier not in carrier_services:
                        carrier_services[carrier] = {}
                    carrier_services[carrier][service_name] = count

            # Format the response by carrier
            for carrier, services in carrier_services.items():
                response += f"**{carrier}**:\n"
                for service_name, count in services.items():
                    response += f"- {service_name}: {count} orders\n"
                response += "\n"

            # Add information about shipping zones
            response += "\n**About Shipping Zones**:\n"
            response += "Our system selects delivery services based on shipping zones. "
            response += (
                "Lower zone numbers (1-2) indicate shorter distances from the fulfillment center, "
            )
            response += "while higher numbers (7-8) indicate longer distances. "
            response += "We have two fulfillment centers: Moorpark (CA) and Wheeling (IL).\n\n"

            # Add information about how delivery services are selected
            response += "**How Delivery Services are Selected**:\n"
            response += "1. We match delivery services based on the first 3 digits of the destination ZIP code\n"
            response += "2. We consider the origin ZIP code of the fulfillment center (93021 for Moorpark, 60090 for Wheeling)\n"
            response += (
                "3. If no exact match is found, we look for services with similar ZIP code ranges\n"
            )
            response += "4. If still no match, we default to UPS Ground with 5 days delivery time\n"

            return response
        else:
            return "I don't see any delivery service information in the processed orders."

    # Handle fulfillment center related queries
    elif any(
        keyword in prompt_lower
        for keyword in ["fulfillment", "center", "warehouse", "moorpark", "wheeling"]
    ):
        if "Fulfillment Center" in processed_orders.columns:
            fc_counts = processed_orders["Fulfillment Center"].value_counts().to_dict()
            response = "Here's a breakdown of orders by fulfillment center:\n\n"
            for fc, count in fc_counts.items():
                if pd.notna(fc) and fc:
                    response += f"- {fc}: {count} orders\n"

            # Add information about fulfillment center selection
            response += "\n**About Fulfillment Centers**:\n"
            response += "We have two fulfillment centers:\n"
            response += "1. **Moorpark (CA)** - ZIP code 93021\n"
            response += "2. **Wheeling (IL)** - ZIP code 60090\n\n"

            response += "**How Fulfillment Centers are Selected**:\n"
            response += "Our system uses shipping zones to optimize fulfillment center selection:\n"
            response += "- Lower zone numbers (1-2) indicate shorter distances from the fulfillment center\n"
            response += "- Higher zone numbers (7-8) indicate longer distances\n"
            response += "- We typically select the fulfillment center with the lower shipping zone for the customer's ZIP code\n"
            response += "- Special rules can override the shipping zone-based selection for priority orders or specific ZIP code patterns\n"

            # If we have ZIP code data, add some analysis
            if (
                "shiptopostalcode" in processed_orders.columns
                and "Fulfillment Center" in processed_orders.columns
            ):
                # Get a sample of ZIP codes for each fulfillment center
                moorpark_zips = (
                    processed_orders[
                        processed_orders["Fulfillment Center"].str.contains(
                            "Moorpark", case=False, na=False
                        )
                    ]["shiptopostalcode"]
                    .sample(min(3, len(processed_orders)))
                    .tolist()
                )
                wheeling_zips = (
                    processed_orders[
                        processed_orders["Fulfillment Center"].str.contains(
                            "Wheeling", case=False, na=False
                        )
                    ]["shiptopostalcode"]
                    .sample(min(3, len(processed_orders)))
                    .tolist()
                )

                if moorpark_zips:
                    response += "\nSample ZIP codes fulfilled by Moorpark (CA):\n"
                    for zip_code in moorpark_zips:
                        if pd.notna(zip_code) and zip_code:
                            response += f"- {zip_code}\n"

                if wheeling_zips:
                    response += "\nSample ZIP codes fulfilled by Wheeling (IL):\n"
                    for zip_code in wheeling_zips:
                        if pd.notna(zip_code) and zip_code:
                            response += f"- {zip_code}\n"

            return response
        else:
            return "I don't see any fulfillment center information in the processed orders."

    # Handle order count or summary queries
    elif any(keyword in prompt_lower for keyword in ["orders", "count", "summary", "total"]):
        total_orders = len(processed_orders)
        unique_orders = (
            processed_orders["externalorderid"].nunique()
            if "externalorderid" in processed_orders.columns
            else "unknown"
        )
        response = f"I found {total_orders} order items in the data, representing {unique_orders} unique orders.\n\n"

        # Add more order summary information if available
        if "shiptostate" in processed_orders.columns:
            state_counts = processed_orders["shiptostate"].value_counts().head(5).to_dict()
            response += "Top shipping states:\n"
            for state, count in state_counts.items():
                if pd.notna(state) and state:
                    response += f"- {state}: {count} orders\n"

        return response

    # Handle ZIP code related queries
    elif any(keyword in prompt_lower for keyword in ["zip", "postal", "code", "location"]):
        if "shiptopostalcode" in processed_orders.columns:
            zip_sample = (
                processed_orders["shiptopostalcode"].sample(min(5, len(processed_orders))).tolist()
            )
            response = "Here are some sample ZIP codes from the orders:\n\n"
            for zip_code in zip_sample:
                if pd.notna(zip_code) and zip_code:
                    response += f"- {zip_code}\n"
            return response
        else:
            return "I don't see any ZIP code information in the processed orders."

    # Default response for other queries
    else:
        return "I can help you analyze order data, delivery services, and fulfillment centers. Please ask me about orders, delivery services, fulfillment centers, or ZIP codes."


def render_rule_editor(rules, bundles, override_log):
    """
    Render rule editor interface

    Args:
        rules: List of rule dictionaries
        bundles: Dictionary of bundles
        override_log: List of override logs

    Returns:
        bool: True if rules were updated, False otherwise
    """
    updated = False

    # Create tabs for different rule types
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Zip Code Rules", "Bundle Rules", "Priority Rules", "Override Log"]
    )

    with tab1:
        st.subheader("Zip Code Rules")
        st.write("These rules determine which fulfillment center to use based on zip code.")

        # Show existing zip rules
        zip_rules = [rule for rule in rules if rule["type"] == "zip"]

        if zip_rules:
            zip_df = pd.DataFrame(zip_rules)

            # Add column configuration for better display
            zip_column_config = {
                "type": st.column_config.SelectboxColumn(
                    "type", help="Rule type", width="small", options=["zip"], required=True
                ),
                "condition": st.column_config.TextColumn(
                    "condition",
                    help="ZIP code condition (e.g., 'starts with 9')",
                    width="medium",
                    required=True,
                ),
                "action": st.column_config.TextColumn(
                    "action",
                    help="Action to take (e.g., 'warehouse = CA-Oxnard-93030')",
                    width="large",
                    required=True,
                ),
            }

            # Using Streamlit's built-in filtering capability

            edited_zip_df = st.data_editor(
                zip_df,
                use_container_width=True,
                num_rows="dynamic",
                key="zip_rules_editor",
                column_config=zip_column_config,
                hide_index=True,
            )

            # Check if rules were updated
            if not zip_df.equals(edited_zip_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "zip"]
                updated_rules.extend(edited_zip_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "zip rules",
                        "old_value": str(zip_df.shape[0]),
                        "new_value": str(edited_zip_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No zip code rules defined yet.")

        # Add new zip rule
        st.subheader("Add New Zip Rule")
        with st.form("add_zip_rule"):
            condition = st.text_input("Condition (e.g., 'starts with 9')")
            action = st.text_input("Action (e.g., 'warehouse = CA-Oxnard-93030')")

            if st.form_submit_button("Add Rule"):
                if condition and action:
                    new_rule = {"type": "zip", "condition": condition, "action": action}

                    # Add to rules
                    st.session_state.rules.append(new_rule)

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "zip rule",
                            "old_value": "",
                            "new_value": f"{condition} -> {action}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Rule added successfully!")
                else:
                    st.error("Please fill in both condition and action.")

    with tab2:
        st.subheader("Bundle Rules")
        st.write("These rules determine which products belong to which bundle.")

        # Show existing bundle rules
        bundle_rules = [rule for rule in rules if rule["type"] == "bundle"]

        if bundle_rules:
            bundle_df = pd.DataFrame(bundle_rules)

            # Add column configuration for better display
            bundle_column_config = {
                "type": st.column_config.SelectboxColumn(
                    "type", help="Rule type", width="small", options=["bundle"], required=True
                ),
                "condition": st.column_config.TextColumn(
                    "condition", help="Bundle condition", width="medium", required=True
                ),
                "action": st.column_config.TextColumn(
                    "action", help="Action to take", width="large", required=True
                ),
            }

            # Using Streamlit's built-in filtering capability

            edited_bundle_df = st.data_editor(
                bundle_df,
                use_container_width=True,
                num_rows="dynamic",
                key="bundle_rules_editor",
                column_config=bundle_column_config,
                hide_index=True,
            )

            # Check if rules were updated
            if not bundle_df.equals(edited_bundle_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "bundle"]
                updated_rules.extend(edited_bundle_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "bundle rules",
                        "old_value": str(bundle_df.shape[0]),
                        "new_value": str(edited_bundle_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No bundle rules defined yet.")

        # Show bundles
        st.subheader("Bundles")

        if bundles:
            bundle_items = []
            for name, fruits in bundles.items():
                bundle_items.append({"Bundle Name": name, "Contents": ", ".join(fruits)})

            bundle_items_df = pd.DataFrame(bundle_items)

            # Add column configuration for better display
            bundle_items_column_config = {
                "Bundle Name": st.column_config.TextColumn(
                    "Bundle Name", help="Name of the bundle", width="medium", required=True
                ),
                "Contents": st.column_config.TextColumn(
                    "Contents",
                    help="Comma-separated list of items in the bundle",
                    width="large",
                    required=True,
                ),
            }

            # Using Streamlit's built-in filtering capability

            edited_bundle_items_df = st.data_editor(
                bundle_items_df,
                use_container_width=True,
                num_rows="dynamic",
                key="bundles_editor",
                column_config=bundle_items_column_config,
                hide_index=True,
            )

            # Check if bundles were updated
            if not bundle_items_df.equals(edited_bundle_items_df):
                # Update bundles
                updated_bundles = {}
                for _, row in edited_bundle_items_df.iterrows():
                    bundle_name = row["Bundle Name"]
                    contents = [item.strip() for item in row["Contents"].split(",")]
                    updated_bundles[bundle_name] = contents

                # Update session state
                st.session_state.bundles = updated_bundles

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "bundles",
                        "old_value": str(len(bundles)),
                        "new_value": str(len(updated_bundles)),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No bundles defined yet.")

        # Add new bundle
        st.subheader("Add New Bundle")
        with st.form("add_bundle"):
            bundle_name = st.text_input("Bundle Name")
            bundle_contents = st.text_input("Contents (comma-separated)")

            if st.form_submit_button("Add Bundle"):
                if bundle_name and bundle_contents:
                    contents = [item.strip() for item in bundle_contents.split(",")]

                    # Add to bundles
                    st.session_state.bundles[bundle_name] = contents

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "bundle",
                            "old_value": "",
                            "new_value": f"{bundle_name}: {bundle_contents}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Bundle added successfully!")
                else:
                    st.error("Please fill in both bundle name and contents.")

    with tab3:
        st.subheader("Priority Rules")
        st.write("These rules determine order priority based on tags or other criteria.")

        # Show existing priority rules
        priority_rules = [rule for rule in rules if rule["type"] == "priority"]

        if priority_rules:
            priority_df = pd.DataFrame(priority_rules)

            # Add column configuration for better display
            priority_column_config = {
                "type": st.column_config.SelectboxColumn(
                    "type", help="Rule type", width="small", options=["priority"], required=True
                ),
                "condition": st.column_config.TextColumn(
                    "condition",
                    help="Priority condition (e.g., 'P1')",
                    width="medium",
                    required=True,
                ),
                "action": st.column_config.TextColumn(
                    "action",
                    help="Action to take (e.g., 'priority = high')",
                    width="large",
                    required=True,
                ),
            }

            # Using Streamlit's built-in filtering capability

            edited_priority_df = st.data_editor(
                priority_df,
                use_container_width=True,
                num_rows="dynamic",
                key="priority_rules_editor",
                column_config=priority_column_config,
                hide_index=True,
            )

            # Check if rules were updated
            if not priority_df.equals(edited_priority_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "priority"]
                updated_rules.extend(edited_priority_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "priority rules",
                        "old_value": str(priority_df.shape[0]),
                        "new_value": str(edited_priority_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No priority rules defined yet.")

        # Add new priority rule
        st.subheader("Add New Priority Rule")
        with st.form("add_priority_rule"):
            condition = st.text_input("Condition (e.g., 'P1')")
            action = st.text_input("Action (e.g., 'priority = high')")

            if st.form_submit_button("Add Rule"):
                if condition and action:
                    new_rule = {"type": "priority", "condition": condition, "action": action}

                    # Add to rules
                    st.session_state.rules.append(new_rule)

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "priority rule",
                            "old_value": "",
                            "new_value": f"{condition} -> {action}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Rule added successfully!")
                else:
                    st.error("Please fill in both condition and action.")

    with tab4:
        st.subheader("Override Log")
        st.write("History of rule changes and overrides.")

        if override_log:
            log_df = pd.DataFrame(override_log)

            # Add column configuration for better display
            log_column_config = {
                "source": st.column_config.TextColumn(
                    "source", help="Source of the override", width="small"
                ),
                "field": st.column_config.TextColumn(
                    "field", help="Field that was changed", width="medium"
                ),
                "old_value": st.column_config.TextColumn(
                    "old_value", help="Previous value", width="medium"
                ),
                "new_value": st.column_config.TextColumn(
                    "new_value", help="New value", width="medium"
                ),
                "reason": st.column_config.TextColumn(
                    "reason", help="Reason for the change", width="medium"
                ),
                "timestamp": st.column_config.DatetimeColumn(
                    "timestamp",
                    help="When the change occurred",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium",
                ),
            }

            # Make the override log editable with built-in filtering
            edited_log_df = st.data_editor(
                log_df,
                use_container_width=True,
                num_rows="dynamic",
                key="override_log_editor",
                column_config=log_column_config,
                hide_index=True,
                disabled=["source", "timestamp"],  # Make some columns read-only
            )

            # Check if log was updated
            if not log_df.equals(edited_log_df):
                # Update the override log
                st.session_state.override_log = edited_log_df.to_dict("records")
                updated = True
        else:
            st.info("No overrides logged yet.")

    return updated
