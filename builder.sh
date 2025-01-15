# Once you have apptainer in your OS, run this script to build the image.
#!/bin/bash

# Set up temporary directory path
TEMP_DIR="/tmp"


# Set environment variables for Apptainer
export APPTAINER_TMPDIR=$TEMP_DIR

# Clean up any existing temporary files to free space
rm -rf $TEMP_DIR/*

# Check available space before building
echo "Checking available space..."
df -h /tmp

# Build the container
echo "Starting container build..."
apptainer build --fakeroot superpoint_dinov2.sif superpoint_dinov2.def 2>&1 | tee build_log.txt