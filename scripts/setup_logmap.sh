#!/bin/bash

# Target directory
TARGET_DIR="./apis/logmap"
JAR_NAME="logmap-matcher.jar"
ZIP_URL="https://github.com/ernestojimenezruiz/logmap-matcher/releases/download/logmap-matcher-july-2021/logmap-matcher-standalone-july-2021.zip"

echo "Creating directory $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

echo "Downloading LogMap standalone zip..."
wget -qO "$TARGET_DIR/logmap.zip" "$ZIP_URL"

echo "Extracting..."
unzip -qo "$TARGET_DIR/logmap.zip" -d "$TARGET_DIR"

echo "Organizing files..."
# The zip extracts into a folder 'logmap-matcher-standalone-july-2021'
mv "$TARGET_DIR"/logmap-matcher-standalone-july-2021/* "$TARGET_DIR"/

# The jar is typically named logmap-matcher-4.0.jar, rename it for simplicity
mv "$TARGET_DIR"/logmap-matcher-*.jar "$TARGET_DIR/$JAR_NAME"

echo "Cleaning up..."
rm -rf "$TARGET_DIR"/logmap-matcher-standalone-july-2021 "$TARGET_DIR"/logmap.zip

echo "LogMap is successfully set up at $TARGET_DIR/$JAR_NAME"

