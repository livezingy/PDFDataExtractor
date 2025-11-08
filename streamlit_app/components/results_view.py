# streamlit_app/components/results_view.py
"""
Results display component
"""

import streamlit as st
import pandas as pd
from io import BytesIO

def render_results_view(processing_state: dict):
    """
    Render results view
    
    Args:
        processing_state: Processing state dictionary
    """
    st.subheader("📊 Extraction Results")
    
    extracted_tables = processing_state.get('extracted_tables', [])
    
    if not extracted_tables:
        st.info("No table data extracted")
        return
    
    # Display statistics
    st.markdown("### 📈 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tables", len(extracted_tables))
    
    with col2:
        total_rows = sum(table.get('rows', 0) for table in extracted_tables)
        st.metric("Total Rows", total_rows)
    
    with col3:
        total_cols = sum(table.get('cols', 0) for table in extracted_tables)
        st.metric("Total Columns", total_cols)
    
    with col4:
        avg_score = sum(table.get('score', 0) for table in extracted_tables) / len(extracted_tables)
        st.metric("Average Score", f"{avg_score:.2f}")
    
    st.markdown("---")
    
    # Display each table
    st.markdown("### 📋 Table Details")
    
    for table in extracted_tables:
        table_id = table.get('id', 0)
        table_data = table.get('data')
        table_score = table.get('score', 0)
        table_method = table.get('method', 'unknown')
        table_flavor = table.get('flavor', 'unknown')
        table_page = table.get('page_num', 0)
        table_rows = table.get('rows', 0)
        table_cols = table.get('cols', 0)
        
        # Table information card
        with st.expander(f"📊 Table {table_id} - Score: {table_score:.2f} | Method: {table_method} ({table_flavor})"):
            # Display table metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Page", table_page)
            
            with col2:
                st.metric("Rows", table_rows)
            
            with col3:
                st.metric("Columns", table_cols)
            
            with col4:
                st.metric("Score", f"{table_score:.2f}")
            
            st.markdown("---")
            
            # Display table data
            if table_data is not None:
                try:
                    # Ensure it's a DataFrame
                    if isinstance(table_data, pd.DataFrame):
                        if not table_data.empty:
                            st.markdown("#### 📄 Table Data")
                            st.dataframe(table_data, use_container_width=True)
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download as CSV
                                csv = table_data.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label=f"📥 Download Table {table_id} (CSV)",
                                    data=csv,
                                    file_name=f"table_{table_id}.csv",
                                    mime="text/csv",
                                    key=f"download_csv_{table_id}"
                                )
                            
                            with col2:
                                # Download as Excel (if available)
                                try:
                                    excel_buffer = BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                        table_data.to_excel(writer, index=False, sheet_name=f'Table_{table_id}')
                                    excel_data = excel_buffer.getvalue()
                                    
                                    st.download_button(
                                        label=f"📥 Download Table {table_id} (Excel)",
                                        data=excel_data,
                                        file_name=f"table_{table_id}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"download_excel_{table_id}"
                                    )
                                except ImportError:
                                    st.info("Excel export requires openpyxl: `pip install openpyxl`")
                        else:
                            st.warning("Table data is empty")
                    else:
                        # Try to convert to DataFrame
                        try:
                            df = pd.DataFrame(table_data)
                            if not df.empty:
                                st.markdown("#### 📄 Table Data")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download buttons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    csv = df.to_csv(index=False).encode('utf-8-sig')
                                    st.download_button(
                                        label=f"📥 Download Table {table_id} (CSV)",
                                        data=csv,
                                        file_name=f"table_{table_id}.csv",
                                        mime="text/csv",
                                        key=f"download_csv_{table_id}"
                                    )
                                
                                with col2:
                                    try:
                                        excel_buffer = BytesIO()
                                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                            df.to_excel(writer, index=False, sheet_name=f'Table_{table_id}')
                                        excel_data = excel_buffer.getvalue()
                                        
                                        st.download_button(
                                            label=f"📥 Download Table {table_id} (Excel)",
                                            data=excel_data,
                                            file_name=f"table_{table_id}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            key=f"download_excel_{table_id}"
                                        )
                                    except ImportError:
                                        st.info("Excel export requires openpyxl: `pip install openpyxl`")
                            else:
                                st.warning("Table data is empty")
                        except Exception as e:
                            st.warning(f"Cannot display table data (type: {type(table_data)}, error: {str(e)})")
                except ImportError:
                    st.warning("pandas library required to display table data")
            else:
                st.warning("Table data is empty")
        
        st.markdown("---")
    
    # Batch download option
    st.markdown("### 📦 Batch Download")
    
    if st.button("📥 Download All Tables (ZIP)", use_container_width=True):
        st.info("Batch download feature under development...")
