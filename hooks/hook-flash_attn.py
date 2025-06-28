# Hook to exclude flash_attn completely
# This prevents transformers from trying to import flash_attn modules

hiddenimports = []
datas = []
binaries = []

# Exclude all flash_attn related modules
excludedimports = [
    'flash_attn',
    'flash_attn.layers',
    'flash_attn.ops',
    'flash_attn.layers.rotary',
    'flash_attn.ops.triton',
    'flash_attn.ops.triton.rotary',
]
