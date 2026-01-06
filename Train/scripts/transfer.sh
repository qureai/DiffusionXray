#!/bin/bash

# Variables
LOCAL_DIRECTORY="/local_storage/aryan_training_data/training1"
REMOTE_USER="aryan.goyal@e2ecloud13.e2e.qure.ai"
REMOTE_HOST="aryan.goyal@e2ecloud18.e2e.qure.ai"
REMOTE_DIRECTORY="/local_storage/aryab_training"

# Rsync command to transfer the directory
rsync -avz $LOCAL_DIRECTORY/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIRECTORY

# Check if the transfer was successful
if [ $? -eq 0 ]; then
  echo "Directory transfer successful"
else
  echo "Directory transfer failed"
fi
