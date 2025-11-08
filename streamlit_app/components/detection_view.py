# streamlit_app/components/detection_view.py
"""
Detection process display component
"""

import streamlit as st

def render_detection_view(processing_state: dict):
    """
    Render detection process view
    
    Args:
        processing_state: Processing state dictionary
    """
    st.subheader("🔍 Detection Process")
    
    detection_steps = processing_state.get('detection_steps', [])
    log_messages = processing_state.get('log_messages', [])
    extraction_params = processing_state.get('extraction_params', {})
    
    # Display log messages (if any)
    if log_messages:
        with st.expander("📋 Processing Log Messages", expanded=False):
            for log_msg in log_messages:
                level = log_msg.get('level', 'INFO')
                message = log_msg.get('message', '')
                
                # Select color based on log level
                if level == 'INFO':
                    st.markdown(f"<span style='color: blue; font-size: 0.85rem;'>ℹ️ {message}</span>", unsafe_allow_html=True)
                elif level == 'WARNING':
                    st.markdown(f"<span style='color: orange; font-size: 0.85rem;'>⚠️ {message}</span>", unsafe_allow_html=True)
                elif level == 'ERROR':
                    st.markdown(f"<span style='color: red; font-size: 0.85rem;'>❌ {message}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='font-size: 0.85rem;'>{message}</span>", unsafe_allow_html=True)
    
    # Display extraction parameters (if any)
    if extraction_params:
        with st.expander("⚙️ Extraction Parameters", expanded=False):
            method = extraction_params.get('method', 'unknown')
            
            if method == 'pdfplumber':
                st.markdown("### PDFPlumber Parameters")
                
                # Default parameters
                st.markdown("#### Default Parameters")
                default_params = extraction_params.get('default_params', {})
                if default_params:
                    st.json(default_params)
                else:
                    st.markdown("""
                    - `snap_tolerance`: 2
                    - `join_tolerance`: 2
                    - `edge_min_length`: 3
                    - `intersection_tolerance`: 3
                    - `min_words_vertical`: 1
                    - `min_words_horizontal`: 1
                    - `text_x_tolerance`: 3
                    - `text_y_tolerance`: 5
                    """)
                
                # Calculated parameters
                st.markdown("#### Calculated Parameters")
                calculated_params = extraction_params.get('calculated_params', {})
                if calculated_params:
                    st.json(calculated_params)
                else:
                    st.info("Parameters calculated based on page features")
            
            elif method == 'camelot':
                flavor = extraction_params.get('flavor', 'unknown')
                
                if flavor == 'lattice':
                    st.markdown("### Camelot Lattice Parameters")
                    
                    # Default parameters
                    st.markdown("#### Default Parameters")
                    default_params = extraction_params.get('default_params', {})
                    if default_params:
                        st.json(default_params)
                    else:
                        st.markdown("""
                        - `flavor`: 'lattice'
                        - `line_scale`: 40
                        - `line_tol`: 2
                        - `joint_tol`: 2
                        """)
                    
                    # Calculated parameters
                    st.markdown("#### Calculated Parameters")
                    calculated_params = extraction_params.get('calculated_params', {})
                    if calculated_params:
                        st.json(calculated_params)
                    else:
                        st.info("Parameters calculated based on page features")
                
                elif flavor == 'stream':
                    st.markdown("### Camelot Stream Parameters")
                    
                    # Default parameters
                    st.markdown("#### Default Parameters")
                    default_params = extraction_params.get('default_params', {})
                    if default_params:
                        st.json(default_params)
                    else:
                        st.markdown("""
                        - `flavor`: 'stream'
                        - `edge_tol`: 50
                        - `row_tol`: 2
                        - `column_tol`: 0
                        """)
                    
                    # Calculated parameters
                    st.markdown("#### Calculated Parameters")
                    calculated_params = extraction_params.get('calculated_params', {})
                    if calculated_params:
                        st.json(calculated_params)
                    else:
                        st.info("Parameters calculated based on page features")
    
    if not detection_steps:
        st.info("No detection process information available")
        return
    
    # Display overall progress with progress bar
    total_steps = len(detection_steps)
    completed_steps = sum(1 for step in detection_steps if step.get('status') == 'success')
    progress = completed_steps / total_steps if total_steps > 0 else 0
    
    st.progress(progress)
    st.caption(f"Progress: {completed_steps}/{total_steps} steps completed ({progress*100:.1f}%)")
    
    st.markdown("---")
    
    # Display detailed information for each step
    for step in detection_steps:
        step_num = step.get('step', 0)
        step_name = step.get('name', 'Unknown Step')
        step_status = step.get('status', 'unknown')
        step_message = step.get('message', '')
        
        # Select icon and color based on status
        if step_status == 'success':
            icon = "✅"
            color = "green"
        elif step_status == 'error':
            icon = "❌"
            color = "red"
        elif step_status == 'processing':
            icon = "⏳"
            color = "blue"
        else:
            icon = "ℹ️"
            color = "gray"
        
        # Display step information (using smaller font)
        st.markdown(f"""
        <div style="padding: 8px; border-left: 4px solid {color}; margin: 3px 0; font-size: 0.9rem;">
            <strong style="font-size: 0.95rem;">{icon} Step {step_num}: {step_name}</strong><br>
            <span style="color: {color}; font-size: 0.85rem;">{step_message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Display statistics
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_count = sum(1 for step in detection_steps if step.get('status') == 'success')
        st.metric("Successful Steps", success_count)
    
    with col2:
        error_count = sum(1 for step in detection_steps if step.get('status') == 'error')
        st.metric("Failed Steps", error_count)
    
    with col3:
        processing_count = sum(1 for step in detection_steps if step.get('status') == 'processing')
        st.metric("Processing", processing_count)
