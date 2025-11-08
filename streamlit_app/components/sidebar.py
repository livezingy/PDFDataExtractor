# streamlit_app/components/sidebar.py
"""
Sidebar component
"""

import streamlit as st
from streamlit_app.streamlit_utils import check_dependencies, MAX_FILE_SIZE_MB
from streamlit_app.components.param_config import render_param_config

def render_sidebar() -> dict:
    """
    Render sidebar
    
    Returns:
        dict: Sidebar configuration
    """
    # Initialize default values
    uploaded_file = None
    method = "PDFPlumber"
    flavor = "auto"
    
    with st.sidebar:
        # Title at the top of sidebar (increased by 1 level from previous)
        st.markdown("""
        <h2 style='font-size: 1.5rem; margin-bottom: 1rem;'>📊 PDF Table Extractor</h2>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Select PDF File",
            type=["pdf"],
            help=f"PDF files only, maximum {MAX_FILE_SIZE_MB} MB"
        )
        
        # Display file size limit notice
        st.info(f"💡 **Test Version Limit**\n\nMaximum file size: **{MAX_FILE_SIZE_MB} MB**")
        
        st.markdown("---")
        
        st.header("⚙️ Extraction Configuration")
        
        # Extraction method selection
        method = st.selectbox(
            "Extraction Method",
            ["PDFPlumber", "Camelot"],
            help="Select table extraction method"
        )
        
        # Flavor selection (changes dynamically based on method)
        if method == "PDFPlumber":
            flavor = st.selectbox(
                "Flavor",
                ["auto", "lines", "text"],
                help="PDFPlumber extraction mode:\n- auto: Auto select\n- lines: For bordered tables\n- text: For unbordered tables"
            )
        elif method == "Camelot":
            flavor = st.selectbox(
                "Flavor",
                ["auto", "lattice", "stream"],
                help="Camelot extraction mode:\n- auto: Auto select\n- lattice: For bordered tables\n- stream: For unbordered tables"
            )
        else:
            flavor = "auto"
        
        # Parameter configuration
        st.markdown("---")
        st.subheader("⚙️ Parameter Configuration")
        
        param_config = render_param_config(method.lower(), flavor.lower() if flavor != "auto" else None)
        
        st.markdown("---")
        
        # Dependency status check
        st.subheader("📦 Dependency Status")
        dependencies = check_dependencies()
        
        for dep_name, available in dependencies.items():
            status = "✅" if available else "❌"
            st.markdown(f"{status} **{dep_name}**")
        
        st.markdown("---")
        
        # Project information
        st.markdown("### 📚 Project Information")
        st.markdown("""
        - **GitHub**: [View Source](https://github.com/livezingy/PDFDataExtractor)
        - **Documentation**: [View Docs](https://github.com/livezingy/PDFDataExtractor/tree/main/docs)
        - **Issue Report**: [Submit Issue](https://github.com/livezingy/PDFDataExtractor/issues)
        """)
        
        st.markdown("---")
        
        # Version information
        st.markdown("### ℹ️ Version Information")
        st.markdown(f"""
        - **Version**: 1.0.0
        - **Test Mode**: Enabled
        - **File Limit**: {MAX_FILE_SIZE_MB} MB
        """)
    
    return {
        'uploaded_file': uploaded_file,
        'method': method,
        'flavor': flavor,
        'param_config': param_config
    }
