# This hook ensures importlib.metadata can find package information
# by including all package metadata directories

hiddenimports = [
    'importlib.metadata',
    'importlib_metadata',
]

datas = []
binaries = []
