from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all torch modules and data
datas, binaries, hiddenimports = collect_all('torch')

# Add specific torch modules that might be missed
hiddenimports += [
    'torch.nn',
    'torch.nn.functional',
    'torch.nn.modules',
    'torch.nn.modules.activation',
    'torch.nn.modules.batchnorm',
    'torch.nn.modules.container',
    'torch.nn.modules.conv',
    'torch.nn.modules.linear',
    'torch.nn.modules.loss',
    'torch.nn.modules.normalization',
    'torch.nn.modules.pooling',
    'torch.nn.modules.rnn',
    'torch.nn.modules.transformer',
    'torch.nn.modules.utils',
    'torch.optim',
    'torch.optim.lr_scheduler',
    'torch.utils',
    'torch.utils.data',
    'torch.utils.data.dataloader',
    'torch.utils.data.dataset',
    'torch.utils.checkpoint',
    'torch.cuda',
    'torch.backends',
    'torch.backends.cuda',
    'torch.backends.cudnn',
    'torch.autograd',
    'torch.autograd.function',
    'torch.jit',
    'torch._C',
    'torch._utils',
]

# Collect torchvision if available
try:
    torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')
    datas += torchvision_datas
    binaries += torchvision_binaries
    hiddenimports += torchvision_hiddenimports
except:
    pass

# Collect NVIDIA CUDA libraries if available
try:
    nvidia_datas, nvidia_binaries, nvidia_hiddenimports = collect_all('nvidia')
    datas += nvidia_datas
    binaries += nvidia_binaries
    hiddenimports += nvidia_hiddenimports
except:
    pass
