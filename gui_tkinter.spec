# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Get the virtual environment path and detect Python version
venv_path = Path('./ocr_env')

# Dynamically find the Python version in the virtual environment
python_version = None
lib_path = venv_path / 'lib'
if lib_path.exists():
    for python_dir in lib_path.iterdir():
        if python_dir.is_dir() and python_dir.name.startswith('python3.'):
            python_version = python_dir.name
            break

if python_version is None:
    # Fallback to system Python version
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

site_packages = venv_path / 'lib' / python_version / 'site-packages'
print(f"Using Python version: {python_version}")
print(f"Site packages path: {site_packages}")

# Define hidden imports for all ML libraries
hiddenimports = [
    # Core ML libraries
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.transforms',
    'transformers',
    'transformers.models',
    'transformers.models.auto',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.processing_utils',
    'accelerate',
    'huggingface_hub',
    'safetensors',
    'tokenizers',
    
    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    
    # PDF processing
    'fitz',
    'PyMuPDF',
    
    # Data processing
    'pandas',
    'numpy',
    'regex',
    
    # NVIDIA CUDA libraries (if present)
    'nvidia',
    'nvidia.cublas',
    'nvidia.cuda_runtime',
    'nvidia.cudnn',
    
    # Other dependencies
    'einops',
    'psutil',
    'tqdm',
    'requests',
    'certifi',
    'charset_normalizer',
    'urllib3',
    'filelock',
    'fsspec',
    'packaging',
    'sympy',
    'mpmath',
    'networkx',
    'jinja2',
    'markupsafe',
    'yaml',
    'dateutil',
    'pytz',
    'six',
    'typing_extensions',
]

# Data files to include
datas = [
    ('batch_ocr.py', '.'),
    ('gui_tkinter.desktop', '.'),
    ('requirements.txt', '.'),
]

# Binary files to include
binaries = []

# Collect package metadata for all dependencies
packages_to_collect = [
    'transformers', 'torch', 'torchvision', 'tokenizers', 'huggingface_hub',
    'accelerate', 'safetensors', 'PIL', 'numpy', 'pandas', 'tqdm', 'requests',
    'certifi', 'charset_normalizer', 'urllib3', 'filelock', 'fsspec',
    'packaging', 'sympy', 'mpmath', 'networkx', 'jinja2', 'markupsafe',
    'yaml', 'dateutil', 'pytz', 'six', 'typing_extensions', 'regex',
    'psutil', 'einops', 'PyMuPDF'
]

# Collect ALL .dist-info and .egg-info directories from site-packages
if site_packages.exists():
    print(f"Scanning for package metadata in: {site_packages}")
    
    # Collect all .dist-info directories
    for dist_info_dir in site_packages.glob('*.dist-info'):
        if dist_info_dir.is_dir():
            datas.append((str(dist_info_dir), dist_info_dir.name))
            print(f"Added dist-info: {dist_info_dir.name}")
    
    # Collect all .egg-info directories
    for egg_info_dir in site_packages.glob('*.egg-info'):
        if egg_info_dir.is_dir():
            datas.append((str(egg_info_dir), egg_info_dir.name))
            print(f"Added egg-info: {egg_info_dir.name}")
        elif egg_info_dir.is_file():  # Some egg-info are files
            datas.append((str(egg_info_dir), egg_info_dir.name))
            print(f"Added egg-info file: {egg_info_dir.name}")

# Collect comprehensive data for major ML libraries
try:
    # Collect transformers
    transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
    datas += transformers_datas
    binaries += transformers_binaries
    hiddenimports += transformers_hiddenimports
    print(f"Collected {len(transformers_datas)} transformers data files")
except Exception as e:
    print(f"Warning: Could not collect transformers data: {e}")

try:
    # Collect torch
    torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
    datas += torch_datas
    binaries += torch_binaries
    hiddenimports += torch_hiddenimports
    print(f"Collected {len(torch_datas)} torch data files")
except Exception as e:
    print(f"Warning: Could not collect torch data: {e}")

try:
    # Collect tokenizers
    tokenizers_datas, tokenizers_binaries, tokenizers_hiddenimports = collect_all('tokenizers')
    datas += tokenizers_datas
    binaries += tokenizers_binaries
    hiddenimports += tokenizers_hiddenimports
    print(f"Collected {len(tokenizers_datas)} tokenizers data files")
except Exception as e:
    print(f"Warning: Could not collect tokenizers data: {e}")

try:
    # Collect huggingface_hub
    hf_datas, hf_binaries, hf_hiddenimports = collect_all('huggingface_hub')
    datas += hf_datas
    binaries += hf_binaries
    hiddenimports += hf_hiddenimports
    print(f"Collected {len(hf_datas)} huggingface_hub data files")
except Exception as e:
    print(f"Warning: Could not collect huggingface_hub data: {e}")

# Collect additional important packages
additional_packages = ['tqdm', 'numpy', 'pandas', 'PIL', 'regex', 'psutil']
for pkg in additional_packages:
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
        print(f"Collected {len(pkg_datas)} {pkg} data files")
    except Exception as e:
        print(f"Warning: Could not collect {pkg} data: {e}")

# Collect additional .so files from the virtual environment
if site_packages.exists():
    for so_file in site_packages.rglob('*.so*'):
        if so_file.is_file() and 'nvidia' in str(so_file):
            rel_path = so_file.relative_to(site_packages)
            binaries.append((str(so_file), str(rel_path.parent)))

block_cipher = None

a = Analysis(
    ['gui_tkinter.py'],
    pathex=[str(site_packages)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'flash_attn',
        'flash_attn.layers',
        'flash_attn.ops',
        'flash_attn.layers.rotary',
        'flash_attn.ops.triton',
        'flash_attn.ops.triton.rotary',
        'triton.runtime.jit',
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
    [],
    exclude_binaries=True,
    name='gui_tkinter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui_tkinter',
)
