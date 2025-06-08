# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules


block_cipher = None

# Add runtime hook to handle Python 3.10 bytecode issues
with open('hook-python310.py', 'w') as f:
    f.write('''
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
''')

# Collect all required packages
# datas, binaries, hiddenimports

datas = []
binaries = []
hiddenimports = []

# Add core packages with specific handling
packages = {
    'sklearn': ['sklearn.base', 'sklearn.utils._cython_blas', 'sklearn.neighbors._partition_nodes', 
                'sklearn.tree._utils', 'sklearn.utils._typedefs', 'sklearn.utils._weight_vector'],
    'torch': ['torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils.data'],
    'torchvision': ['torchvision.transforms', 'torchvision.models'],
    'transformers': ['transformers.models.table_transformer'],
    'camelot': [],
    'pytesseract': [],
    'PIL': [],
    'cv2': [],
    'numpy': []
}

for package, submodules in packages.items():
    try:
        # Collect main package
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
        
        # Add submodules
        hiddenimports.extend(submodules)
        
        # Collect submodules recursively
        if package in ['sklearn', 'torch', 'transformers']:
            hiddenimports.extend(collect_submodules(package))
    except Exception as e:
        print(f"Warning: Failed to collect {package}: {str(e)}")

# Add project directories and static files
# (src, dest) tuples
# models/ 目录下的所有内容（包括Tesseract-OCR和table-transformer）
datas += [
    ('models', 'models'),
    ('core', 'core'),
    ('config', 'config'),
    ('gui', 'gui'),
    ('README.md', '.'),
    ('requirements.txt', '.'),
    ('config/styles.qss', 'config'),
]

# 处理 Tesseract-OCR 可执行文件和 DLL
# 如果 models/Tesseract-OCR 下有 exe/dll，PyInstaller 会自动递归打包 models 目录

# 主程序入口
entry_script = 'main.py'

# Analysis

a = Analysis(
    [entry_script],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=['hook-python310.py'],
    excludes=[
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2',  'tensorboard'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDFDataExtractor',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)