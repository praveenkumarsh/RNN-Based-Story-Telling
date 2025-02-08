FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 80

# Run the Streamlit app on port 80
CMD ["STREAMLIT_EMAIL_ADDRESS=''", "streamlit", "run", "app.py", "--server.port=80", "--server.enableCORS=false"]