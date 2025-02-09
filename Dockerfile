FROM python:3.9-slim

RUN apt-get update && apt-get install -y awscli

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

RUN aws s3 cp s3://story-generate-rnn/data/ data/ --recursive
RUN aws s3 cp s3://story-generate-rnn/models/ models/ --recursive

# Set environment variable to skip email prompt
ENV STREAMLIT_EMAIL_ADDRESS=""


# Expose the port Streamlit runs on
EXPOSE 80

# Run the Streamlit app on port 80
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.enableCORS=false"]