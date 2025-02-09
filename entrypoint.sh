#!/bin/sh

echo "Setting up environment..."

# Ensure AWS credentials are available
aws configure list

# Download models from S3
echo "Downloading models from S3..."
aws s3 cp s3://story-generate-rnn/data/ /app/data/ --recursive
aws s3 cp s3://story-generate-rnn/models/ /app/models/ --recursive

# Start the application
echo "Starting application..."
exec "$@"
