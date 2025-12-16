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
            "Select PDF or Image File",
            type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            help=f"PDF or Image files, maximum {MAX_FILE_SIZE_MB} MB"
        )
        
        # Display file size limit notice
        st.info(f"💡 **Test Version Limit**\n\nMaximum file size: **{MAX_FILE_SIZE_MB} MB**")
        
        st.markdown("---")
        
        st.header("⚙️ Extraction Configuration")
        
        # 根据文件类型显示不同的选择选项
        is_image_file = False
        if uploaded_file is not None:
            file_type = uploaded_file.type.lower()
            is_image_file = file_type.startswith('image/') or uploaded_file.name.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            )
        
        if is_image_file:
            # 检测是否在Streamlit Cloud环境
            import os
            is_streamlit_cloud = os.environ.get('STREAMLIT_CLOUD', '').lower() == 'true' or \
                                'STREAMLIT_SHARING' in os.environ or \
                                os.path.exists('/home/appuser')
            
            # 图像文件：选择检测引擎
            if is_streamlit_cloud:
                # Streamlit Cloud环境：不支持图像表格检测（PaddleOCR/Transformer模型过大）
                st.error("""
                ❌ **Streamlit Cloud 限制**：
                
                **图像表格检测功能（PaddleOCR+PP-Structure / Transformer）在 Streamlit Cloud 上不可用**。
                
                原因：
                - PaddleOCR+PP-Structure 需要下载多个大模型（200-500MB+）
                - Streamlit Cloud 有严格的运行时间和内存限制
                - 模型下载和加载会频繁超时或失败
                
                **解决方案**：
                - 对于图像文件，请在**本地或服务器部署**以使用 PaddleOCR+PP-Structure 或 Transformer
                - 对于 PDF 文件，可以使用 PDFPlumber 或 Camelot（在云端可用）
                """)
                st.markdown("""
                <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    <strong>💡 本地部署指南:</strong> 要使用图像表格检测功能，请参考 
                    <a href='https://github.com/livezingy/PDFDataExtractor/blob/main/docs/deployment_guide.md' target='_blank'>部署指南</a> 
                    在本地或服务器部署。
                </div>
                """, unsafe_allow_html=True)
                # 不设置method，让用户知道图像处理在云端不可用
                method = None
            else:
                # 本地环境：提供两个选项
                method = st.selectbox(
                    "Detection Engine",
                    ["PaddleOCR", "Transformer"],
                    index=0,  # 默认PaddleOCR
                    help="Select table detection engine for image files:\n"
                         "- PaddleOCR: Recommended for Chinese documents, faster, with HTML output\n"
                         "- Transformer: Available only in local deployment, may be more accurate for complex tables"
                )
                
                # 显示引擎说明
                if method == "PaddleOCR":
                    st.info("💡 **PaddleOCR**: Best for Chinese documents, faster processing, supports HTML output")
                    st.warning("""
                    ⚠️ **First-time Use Notice**: 
                    
                    On first use, PaddleOCR will download model files (200-500MB), which may take **2-5 minutes**. 
                    Please be patient and do not close the page. 
                    
                    If you encounter a timeout error, please wait a few minutes and try again, or use PDFPlumber/Camelot for PDF files instead.
                    """)
                else:
                    st.warning("⚠️ **Transformer**: Requires local deployment with sufficient resources. Not available in Streamlit Cloud.")
            
            flavor = None  # 图像文件不需要flavor
        else:
            # PDF文件：选择提取方法
            method = st.selectbox(
                "Extraction Method",
                ["PDFPlumber", "Camelot"],
                index=0,  # 默认PDFPlumber
                help="Select table extraction method for PDF files"
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
        
        # Parameter configuration (仅PDF文件显示)
        if not is_image_file:
            st.markdown("---")
            st.subheader("⚙️ Parameter Configuration")
            param_config = render_param_config(method.lower(), flavor.lower() if flavor and flavor != "auto" else None)
        else:
            param_config = None
        
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
