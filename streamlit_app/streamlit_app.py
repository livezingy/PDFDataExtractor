# streamlit_app/streamlit_app.py
"""
Streamlit application main entry point
"""

# Set environment variable before importing streamlit to disable file watcher
# This avoids asyncio event loop errors when Streamlit checks torch module paths
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# 在应用启动的最早阶段设置环境变量，避免在无头环境中加载OpenGL库
# 这对于Camelot和OpenCV等依赖系统库的包非常重要
# 必须在导入任何可能使用这些库的模块之前设置
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')
# Ghostscript设备设置（Camelot依赖Ghostscript）
os.environ.setdefault('GS_DEVICE', 'display')

import streamlit as st
import sys
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.file_info import render_file_info
from streamlit_app.components.detection_view import render_detection_view
from streamlit_app.components.results_view import render_results_view
from streamlit_app.streamlit_utils import (
    save_uploaded_file,
    process_pdf_streamlit,
    cleanup_temp_file,
    MAX_FILE_SIZE_MB,
    MAX_FILE_SIZE
)

# Set page configuration
st.set_page_config(
    page_title="PDF Table Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS styles to adjust font size
st.markdown("""
<style>
    /* Reduce overall font size */
    .main {
        font-size: 0.9rem;
    }
    
    /* Title fonts */
    h1 {
        font-size: 2rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    h3 {
        font-size: 1.2rem;
    }
    
    /* Body fonts */
    p, div, span {
        font-size: 0.9rem;
    }
    
    /* Table fonts */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Code block fonts */
    code {
        font-size: 0.85rem;
    }
    
    /* Sidebar fonts */
    .sidebar .sidebar-content {
        font-size: 0.9rem;
    }
    
    /* Step information fonts */
    div[data-testid="stMarkdownContainer"] {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def check_file_size(file):
    """Check if file size meets the limit"""
    if file.size > MAX_FILE_SIZE:
        file_size_mb = file.size / 1024 / 1024
        st.error(
            f"❌ **File Too Large** ({file_size_mb:.2f} MB)\n\n"
            f"Current version only supports test files (maximum {MAX_FILE_SIZE_MB} MB)\n\n"
            f"Please upload a smaller PDF file for testing."
        )
        return False
    return True

# 在您的 Streamlit 应用中添加这个诊断部分
import streamlit as st
import sys
import subprocess
import pkg_resources

def detailed_diagnosis():
    st.header("详细环境诊断")
    
    # 1. 检查 Python 路径和版本
    st.subheader("Python 环境")
    st.write(f"Python 可执行文件: {sys.executable}")
    st.write(f"Python 版本: {sys.version}")
    st.write(f"路径: {sys.path}")
    
    # 2. 检查所有已安装的包
    st.subheader("所有已安装的包")
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # 查找所有计算机视觉相关的包
    cv_related = {}
    for pkg, version in installed_packages.items():
        if any(keyword in pkg.lower() for keyword in ['cv', 'opencv', 'vision', 'image']):
            cv_related[pkg] = version
    
    if cv_related:
        st.write("计算机视觉相关包:")
        for pkg, version in cv_related.items():
            st.write(f"- {pkg}: {version}")
    else:
        st.warning("未找到明显的计算机视觉相关包")
    
    # 3. 专门检查 OpenCV
    st.subheader("OpenCV 详细检查")
    try:
        import cv2
        st.success(f"✅ OpenCV 导入成功")
        st.write(f"版本: {cv2.__version__}")
        st.write(f"文件路径: {cv2.__file__}")
        
        # 检查构建信息
        try:
            build_info = cv2.getBuildInformation()
            st.write("构建信息（前500字符）:")
            st.text(build_info[:500] + "..." if len(build_info) > 500 else build_info)
        except:
            st.write("无法获取构建信息")
            
    except ImportError as e:
        st.error(f"❌ OpenCV 导入失败: {e}")
        
    # 4. 检查 camelot
    st.subheader("Camelot 检查")
    # 在导入camelot之前设置环境变量，避免在无头环境中加载OpenGL库
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('DISPLAY', '')
    try:
        import camelot
        st.success(f"✅ Camelot 导入成功 - 版本: {camelot.__version__}")
    except ImportError as e:
        st.error(f"❌ Camelot 导入失败: {e}")
    except Exception as e:
        st.error(f"❌ Camelot 检查出错: {e}")

# 在侧边栏添加快速诊断
def sidebar_quick_check():
    st.sidebar.header("快速检查")
    
    # OpenCV 检查
    try:
        import cv2
        st.sidebar.success(f"OpenCV: {cv2.__version__}")
        
        # 检查是否是 headless
        if 'headless' in cv2.__file__.lower():
            st.sidebar.success("Headless 版本")
        else:
            st.sidebar.warning("可能是 GUI 版本")
            
    except ImportError as e:
        st.sidebar.error(f"OpenCV 失败: {e}")
    
    # Camelot 检查
    # 在导入camelot之前设置环境变量，避免在无头环境中加载OpenGL库
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('DISPLAY', '')
    try:
        import camelot
        st.sidebar.success(f"Camelot: {camelot.__version__}")
    except Exception as e:
        st.sidebar.error(f"Camelot 失败: {e}")



def main():
    """Main function"""
    # detailed_diagnosis()
    # Initialize session_state
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = {
            'file_info': None,
            'detection_steps': [],
            'extracted_tables': [],
            'visualizations': {},
            'temp_file_path': None,
            'extraction_params': {}
        }
    
    # Render sidebar
    sidebar_config = render_sidebar()
    
    # Main interface
    uploaded_file = sidebar_config.get('uploaded_file')
    
    if uploaded_file is not None:
        # Check file size
        if not check_file_size(uploaded_file):
            st.stop()
        
        # Display file information
        render_file_info(uploaded_file)
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
            # Get configuration
            method = sidebar_config.get('method', 'PDFPlumber').lower()
            flavor = sidebar_config.get('flavor', 'auto')
            # Use default score threshold (0.6)
            score_threshold = 0.6
            
            # Processing parameters
            params = {
                'table_method': method,
                'table_flavor': flavor,
                'table_score_threshold': score_threshold,
                'pages': 'all'
            }
            
            # Process file
            with st.spinner("Processing PDF file, please wait..."):
                try:
                    # Save uploaded file to temporary directory
                    temp_file_path = save_uploaded_file(uploaded_file)
                    st.session_state.processing_state['temp_file_path'] = temp_file_path
                    
                    # Process PDF
                    results = process_pdf_streamlit(
                        pdf_path=temp_file_path,
                        method=method,
                        flavor=flavor,
                        params=params,
                        param_config=sidebar_config.get('param_config')
                    )
                    
                    # Update session_state
                    st.session_state.processing_state.update(results)
                    
                    st.success("✅ Processing completed!")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    with st.expander("View detailed error information"):
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display detection process (if data available)
        if st.session_state.processing_state.get('detection_steps'):
            st.markdown("---")
            render_detection_view(st.session_state.processing_state)
        
        # Display extraction results (if data available)
        if st.session_state.processing_state.get('extracted_tables'):
            st.markdown("---")
            render_results_view(st.session_state.processing_state)
        
        # Clean up temporary files (at end of session)
        if st.session_state.processing_state.get('temp_file_path'):
            if st.button("🧹 Clean Up Temporary Files"):
                cleanup_temp_file(st.session_state.processing_state['temp_file_path'])
                st.session_state.processing_state['temp_file_path'] = None
                st.success("✅ Temporary files cleaned up")
    else:
        # Welcome interface
        st.info("👈 Please upload a PDF file from the left sidebar to get started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ## 🎯 Feature Description
            
            ### Supported Extraction Methods
            
            - **PDFPlumber**: Heuristic extraction based on text and lines
              - `lines`: For bordered tables
              - `text`: For unbordered tables
              - `auto`: Automatically select best mode
            
            - **Camelot**: Structure extraction based on machine learning
              - `lattice`: For bordered tables
              - `stream`: For unbordered tables
              - `auto`: Automatically select best mode
            
            ### 📊 Processing Workflow
            
            1. **File Upload**: Upload PDF file (maximum 10 MB)
            2. **Feature Analysis**: Analyze page features (lines, text, characters)
            3. **Table Type Detection**: Detect bordered/unbordered tables
            4. **Parameter Calculation**: Automatically calculate optimal parameters
            5. **Table Extraction**: Extract tables using selected method
            6. **Result Display**: Display extracted table data
            """)
        
        with col2:
            st.markdown("""
            ## ⚠️ Test Version Limitations
            
            ### File Size Limit
            
            - Current version only supports test files
            - **Maximum file size**: 10 MB
            - Recommend using small PDF files for testing
            
            ### Feature Limitations
            
            - Transformer models are not available (large model files). Download the repository and model files to local PC could test the Transformer Feature.
            - Recommend using PDFPlumber method first
            
            ### Usage Recommendations
            
            1. Use small PDF files (< 10 MB)
            2. Prefer PDFPlumber method
            3. If issues occur, check error messages
            
            ## 📚 Related Links
            
            - [GitHub Repository](https://github.com/livezingy/PDFDataExtractor)
            - [Technical Documentation](https://github.com/livezingy/PDFDataExtractor/tree/main/docs)
            - [Issue Report](https://github.com/livezingy/PDFDataExtractor/issues)
            """)

if __name__ == "__main__":
    main()
