# streamlit_app/components/param_config.py
"""
Parameter Configuration Component for Streamlit

Provides UI for configuring PDFPlumber and Camelot extraction parameters
with three modes: default, auto, and custom.
"""

import streamlit as st
from core.utils.param_config import (
    PDFPLUMBER_PARAM_DEFS, CAMELOT_LATTICE_PARAM_DEFS, CAMELOT_STREAM_PARAM_DEFS,
    validate_pdfplumber_params, validate_camelot_lattice_params, validate_camelot_stream_params,
    get_default_pdfplumber_params, get_default_camelot_lattice_params, get_default_camelot_stream_params
)


def render_param_config(method: str, flavor: str = None) -> dict:
    """
    Render parameter configuration UI
    
    Args:
        method: Extraction method ('pdfplumber' or 'camelot')
        flavor: Extraction flavor ('lines', 'text', 'lattice', 'stream')
        
    Returns:
        dict: Parameter configuration with 'mode' and 'params' keys
    """
    method = method.lower()
    flavor = flavor.lower() if flavor else None
    
    # Get parameter definitions
    if method == 'pdfplumber':
        param_defs = PDFPLUMBER_PARAM_DEFS
    elif method == 'camelot':
        if flavor == 'lattice':
            param_defs = CAMELOT_LATTICE_PARAM_DEFS
        elif flavor == 'stream':
            param_defs = CAMELOT_STREAM_PARAM_DEFS
        else:
            param_defs = {}
    else:
        param_defs = {}
    
    # Parameter mode selection
    # If flavor is None (auto), disable Custom option
    mode_options = ["Default", "Auto", "Custom"]
    default_index = 1  # Default to Auto
    
    # If flavor is None (auto), remove Custom from options
    if flavor is None:
        mode_options = ["Default", "Auto"]
        # If current selection was Custom, it will be reset to Auto
        default_index = 1
    
    mode = st.selectbox(
        "Parameter Mode",
        mode_options,
        index=default_index,
        key=f"param_mode_{method}_{flavor or 'none'}"
    )
    mode = mode.lower()
    
    params = {}
    
    if mode == 'default':
        if method == 'pdfplumber':
            params = get_default_pdfplumber_params()
        elif method == 'camelot':
            if flavor == 'lattice':
                params = get_default_camelot_lattice_params()
            elif flavor == 'stream':
                params = get_default_camelot_stream_params()
    elif mode == 'auto':
        params = {}  # Auto mode uses calculated parameters
    else:  # custom
        st.subheader("Custom Parameters")
        
        # Check if flavor is required for custom parameters
        if method == 'camelot' and flavor is None:
            st.warning(
                "⚠️ **Flavor Required for Custom Parameters**\n\n"
                "To configure custom parameters for Camelot, please first select a specific Flavor (Lattice or Stream) instead of 'Auto'.\n\n"
                "Custom parameters are flavor-specific and cannot be configured when Flavor is set to 'Auto'."
            )
            params = {}  # Return empty params when flavor is not specified
        elif method == 'pdfplumber' and flavor is None:
            st.warning(
                "⚠️ **Flavor Required for Custom Parameters**\n\n"
                "To configure custom parameters for PDFPlumber, please first select a specific Flavor (Lines or Text) instead of 'Auto'.\n\n"
                "Custom parameters are flavor-specific and cannot be configured when Flavor is set to 'Auto'."
            )
            params = {}  # Return empty params when flavor is not specified
        elif not param_defs:
            st.warning(
                "⚠️ **No Parameters Available**\n\n"
                "No custom parameters are available for the current method and flavor combination."
            )
            params = {}  # Return empty params when no definitions available
        else:
            # Create parameter inputs
            for param_name, param_def in param_defs.items():
                param_key = f"param_{method}_{flavor or 'none'}_{param_name}"
                
                label = param_name.replace('_', ' ').title()
                tooltip = param_def.get('description', '')
                
                if param_def['type'] == int:
                    value = st.number_input(
                        label,
                        min_value=int(param_def['range'][0]) if 'range' in param_def else None,
                        max_value=int(param_def['range'][1]) if 'range' in param_def else None,
                        value=int(param_def['default']),
                        help=tooltip,
                        key=param_key
                    )
                    params[param_name] = value
                elif param_def['type'] == float:
                    value = st.number_input(
                        label,
                        min_value=float(param_def['range'][0]) if 'range' in param_def else None,
                        max_value=float(param_def['range'][1]) if 'range' in param_def else None,
                        value=float(param_def['default']),
                        step=0.1,
                        help=tooltip,
                        key=param_key
                    )
                    params[param_name] = value
                elif param_def['type'] == str:
                    if 'options' in param_def:
                        value = st.selectbox(
                            label,
                            param_def['options'],
                            index=param_def['options'].index(param_def['default']) if param_def['default'] in param_def['options'] else 0,
                            help=tooltip,
                            key=param_key
                        )
                        params[param_name] = value
                    else:
                        value = st.text_input(
                            label,
                            value=str(param_def['default']),
                            help=tooltip,
                            key=param_key
                        )
                        params[param_name] = value
            
            # For Camelot, ensure flavor is included in custom params
            if method == 'camelot' and flavor:
                params['flavor'] = flavor
        
        # Validate parameters (only if params were collected)
        if params and param_defs:
            # Store flavor before validation (validation functions may remove it)
            camelot_flavor = None
            if method == 'camelot' and flavor and 'flavor' in params:
                camelot_flavor = params.get('flavor', flavor)
            
            if method == 'pdfplumber':
                is_valid, error_msg, validated_params = validate_pdfplumber_params(params)
            elif method == 'camelot':
                if flavor == 'lattice':
                    is_valid, error_msg, validated_params = validate_camelot_lattice_params(params)
                elif flavor == 'stream':
                    is_valid, error_msg, validated_params = validate_camelot_stream_params(params)
                else:
                    is_valid, error_msg, validated_params = True, None, params
            else:
                is_valid, error_msg, validated_params = True, None, params
            
            # Restore flavor after validation for Camelot
            if method == 'camelot' and camelot_flavor:
                validated_params['flavor'] = camelot_flavor
            
            if not is_valid and error_msg:
                st.error(f"Parameter validation error: {error_msg}")
                params = validated_params
    
    return {
        'mode': mode,
        'params': params
    }

