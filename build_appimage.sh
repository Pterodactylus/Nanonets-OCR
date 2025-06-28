#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

APP_NAME="gui_tkinter" # Changed to match the main script name
APP_DIR="${SCRIPT_DIR}/dist/${APP_NAME}"
LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_PATH="${SCRIPT_DIR}/linuxdeploy-x86_64.AppImage"

# --- Step 0: Pre-requisites (PyInstaller) ---
echo "--- Checking for PyInstaller ---"
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyInstaller. Please install it manually (pip install pyinstaller) and try again."
        exit 1
    fi
fi

# --- Step 2: Run PyInstaller ---
echo "--- Running PyInstaller ---"
# Run PyInstaller to create a one-directory bundle
pyinstaller --noconfirm --onedir gui_tkinter.py --name "${APP_NAME}" \
    --add-data "batch_ocr.py:." \
    --add-data "gui_tkinter.desktop:."
if [ $? -ne 0 ]; then
    echo "Error: PyInstaller failed. Check the output above for details."
    exit 1
fi

# --- Step 3: Download linuxdeploy if not present ---
echo "--- Checking for linuxdeploy ---"
if [ ! -f "$LINUXDEPLOY_PATH" ]; then
    echo "linuxdeploy not found. Downloading..."
    wget -O "$LINUXDEPLOY_PATH" "$LINUXDEPLOY_URL"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download linuxdeploy. Please download it manually from $LINUXDEPLOY_URL and place it in the current directory."
        exit 1
    fi
    chmod +x "$LINUXDEPLOY_PATH"
fi

# --- Step 4: Prepare AppDir structure ---
echo "--- Preparing AppDir structure ---"
APPDIR_ROOT="${SCRIPT_DIR}/AppDir"
APPDIR_BIN="${APPDIR_ROOT}/usr/bin"
APPDIR_DESKTOP="${APPDIR_ROOT}/usr/share/applications"
APPDIR_ICONS="${APPDIR_ROOT}/usr/share/icons/hicolor/256x256/apps"

# Clean up previous AppDir if it exists
rm -rf "$APPDIR_ROOT"

mkdir -p "$APPDIR_BIN"
mkdir -p "$APPDIR_DESKTOP"
mkdir -p "$APPDIR_ICONS"

echo "--- Contents of ${APP_DIR} before move ---"
ls -l "${APP_DIR}"
echo "--- Contents of ${APP_DIR}/* before move ---"
ls -l "${APP_DIR}"/*

# Move PyInstaller output into AppDir/usr/bin
mv "${APP_DIR}"/* "$APPDIR_BIN"/
if [ $? -ne 0 ]; then
    echo "Error: Failed to move PyInstaller output to AppDir."
    exit 1
fi

# Copy desktop file into AppDir/usr/share/applications
cp "gui_tkinter.desktop" "$APPDIR_DESKTOP"/
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy desktop file to AppDir."
    exit 1
fi

# Copy icon file into AppDir/usr/share/icons/hicolor/256x256/apps
cp "assets/favicon-32x32.png" "${APPDIR_ICONS}/gui_tkinter.png"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy icon file to AppDir."
    exit 1
fi

# --- Step 5: Create AppImage ---
echo "--- Creating AppImage ---"
"$LINUXDEPLOY_PATH" \
    --appdir "$APPDIR_ROOT" \
    --output appimage \
    --executable "${APPDIR_BIN}/${APP_NAME}" \
    --desktop-file "${APPDIR_DESKTOP}/gui_tkinter.desktop" \
    --icon-file "${APPDIR_ICONS}/gui_tkinter.png"

if [ $? -ne 0 ]; then
    echo "Error: linuxdeploy failed. Check the output above for details."
    exit 1
fi

echo "--- AppImage creation complete! ---"
echo "Your AppImage should be in the current directory."
echo "You can run it with: ./${APP_NAME}.AppImage"
