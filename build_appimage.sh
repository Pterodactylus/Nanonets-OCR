#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

APP_NAME="gui_tkinter"
APP_DIR="${SCRIPT_DIR}/dist/${APP_NAME}"
LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_PATH="${SCRIPT_DIR}/linuxdeploy-x86_64.AppImage"
APPIMAGE_OUTPUT_DIR="${SCRIPT_DIR}/appimage_build"
VENV_PATH="${SCRIPT_DIR}/ocr_env"

# --- Step 0: Check virtual environment ---
echo "--- Checking virtual environment ---"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create and activate the virtual environment with all dependencies installed."
    exit 1
fi

# Activate virtual environment
echo "--- Activating virtual environment ---"
source "${VENV_PATH}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# --- Step 1: Check PyInstaller ---
echo "--- Checking for PyInstaller ---"
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstaller not found in virtual environment. Installing..."
    pip install pyinstaller
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyInstaller. Please install it manually and try again."
        exit 1
    fi
fi

# --- Step 2: Clean previous builds ---
echo "--- Cleaning previous builds ---"
rm -rf "${SCRIPT_DIR}/build"
rm -rf "${SCRIPT_DIR}/dist"
rm -rf "${SCRIPT_DIR}/AppDir"

# --- Step 3: Run PyInstaller with spec file ---
echo "--- Running PyInstaller with enhanced configuration ---"
echo "This may take several minutes due to the large ML dependencies..."

# Set environment variables for better PyInstaller performance
export PYTHONOPTIMIZE=1
export TRANSFORMERS_VERBOSITY=error

# Run PyInstaller using the spec file
pyinstaller --noconfirm "${SCRIPT_DIR}/gui_tkinter.spec"

if [ $? -ne 0 ]; then
    echo "Error: PyInstaller failed. Check the output above for details."
    exit 1
fi

echo "--- PyInstaller completed successfully ---"
echo "Bundle size: $(du -sh "${APP_DIR}" | cut -f1)"

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
cp "${SCRIPT_DIR}/gui_tkinter.desktop" "$APPDIR_DESKTOP"/
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy desktop file to AppDir."
    exit 1
fi

# Copy icon file into AppDir/usr/share/icons/hicolor/256x256/apps
cp "${SCRIPT_DIR}/assets/favicon-32x32.png" "${APPDIR_ICONS}/gui_tkinter.png"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy icon file to AppDir."
    exit 1
fi

# --- Step 5: Create AppImage ---
echo "--- Creating AppImage ---"
mkdir -p "$APPIMAGE_OUTPUT_DIR"
(
    cd "$APPIMAGE_OUTPUT_DIR"
    "$LINUXDEPLOY_PATH" \
        --appdir "$APPDIR_ROOT" \
        --output appimage \
        --executable "${APPDIR_BIN}/${APP_NAME}" \
        --desktop-file "${SCRIPT_DIR}/gui_tkinter.desktop" \
        --icon-file "${SCRIPT_DIR}/assets/favicon-32x32.png"
)

if [ $? -ne 0 ]; then
    echo "Error: linuxdeploy failed. Check the output above for details."
    exit 1
fi

echo "--- AppImage creation complete! ---"
echo "Your AppImage should be in: ${APPIMAGE_OUTPUT_DIR}"
echo "You can run it with: ${APPIMAGE_OUTPUT_DIR}/${APP_NAME}.AppImage"
