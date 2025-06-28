from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all transformers modules and data
datas, binaries, hiddenimports = collect_all('transformers')

# Add specific transformers modules that might be missed
hiddenimports += [
    'transformers.models.auto.modeling_auto',
    'transformers.models.auto.tokenization_auto',
    'transformers.models.auto.processing_auto',
    'transformers.models.auto.configuration_auto',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.processing_utils',
    'transformers.image_processing_utils',
    'transformers.feature_extraction_utils',
    'transformers.utils',
    'transformers.utils.hub',
    'transformers.utils.logging',
    'transformers.generation',
    'transformers.generation.utils',
]

# Collect tokenizers data
try:
    tokenizers_datas, _, tokenizers_hiddenimports = collect_all('tokenizers')
    datas += tokenizers_datas
    hiddenimports += tokenizers_hiddenimports
except:
    pass
