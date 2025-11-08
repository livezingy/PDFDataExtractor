# streamlit_app/components/file_info.py
"""
File information display component
"""

import streamlit as st
from datetime import datetime

def render_file_info(uploaded_file):
    """
    Render file information
    
    Args:
        uploaded_file: Streamlit uploaded file object
    """
    st.subheader("📄 File Information")
    
    # Calculate file size
    file_size_mb = uploaded_file.size / 1024 / 1024
    file_size_kb = uploaded_file.size / 1024
    
    # Use column layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File Name", uploaded_file.name)
    
    with col2:
        if file_size_mb >= 1:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        else:
            st.metric("File Size", f"{file_size_kb:.2f} KB")
    
    with col3:
        st.metric("File Type", uploaded_file.type or "PDF")
    
    with col4:
        # Display file size percentage (relative to maximum limit)
        from streamlit_app.streamlit_utils import MAX_FILE_SIZE_MB, MAX_FILE_SIZE
        size_percentage = (uploaded_file.size / MAX_FILE_SIZE) * 100
        st.metric("Usage", f"{size_percentage:.1f}%")
        
        # Display progress bar
        st.progress(size_percentage / 100)
    
    # Display detailed file information (expandable)
    with st.expander("📋 Detailed Information"):
        st.markdown(f"""
        - **File Name**: `{uploaded_file.name}`
        - **File Size**: {file_size_mb:.2f} MB ({uploaded_file.size:,} bytes)
        - **File Type**: {uploaded_file.type or 'PDF'}
        - **Upload Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Maximum Limit**: {MAX_FILE_SIZE_MB} MB
        """)
        
        # File size warning
        if file_size_mb > MAX_FILE_SIZE_MB * 0.8:
            st.warning(f"⚠️ File size is close to limit ({size_percentage:.1f}%), please use a smaller file")
