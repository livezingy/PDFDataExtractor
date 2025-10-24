
import sys
import os
import importlib.util
import types
import dis

def _patch_dis():
    original_get_instructions = dis.get_instructions
    original_get_instructions_bytes = dis._get_instructions_bytes
    
    def patched_get_instructions_bytes(code, varnames=None, names=None, constants=None, cells=None, linestarts=None, line_offset=0):
        try:
            return original_get_instructions_bytes(code, varnames, names, constants, cells, linestarts, line_offset)
        except IndexError:
            # Return empty iterator on error
            return iter([])
    
    def patched_get_instructions(code, first_line=None):
        try:
            return original_get_instructions(code, first_line)
        except IndexError:
            # Return empty iterator on error
            return iter([])
    
    dis._get_instructions_bytes = patched_get_instructions_bytes
    dis.get_instructions = patched_get_instructions

# Apply patches
if sys.version_info >= (3, 10):
    _patch_dis()
    os.environ['PYTHONOPTIMIZE'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
