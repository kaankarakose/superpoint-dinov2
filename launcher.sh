#Once image is created, run this for launch the container
#!/bin/bash

# Define variables
export INSTANCE_NAME="superpoint_dinov2"
export SIF_FILE="superpoint_dinov2.sif"

# Stop the instance if it's already running
apptainer instance stop $INSTANCE_NAME 2>/dev/null

# Start the Apptainer instance with bindings
apptainer instance start --nv $SIF_FILE $INSTANCE_NAME

# Shell into the instance
apptainer shell --nv instance://$INSTANCE_NAME