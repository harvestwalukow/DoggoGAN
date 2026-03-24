# Use a Python base image that supports PyTorch
FROM python:3.10-slim-bullseye

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend (Node) dependencies and install
COPY infinite-canvas-image-generation/server/package*.json ./server/
WORKDIR /app/server
RUN npm install

# Copy all project files
WORKDIR /app
COPY dog_gan_64.pth .
COPY inference.py .
COPY train_code.py .
COPY infinite-canvas-image-generation/server ./server

# Build the TypeScript server
WORKDIR /app/server
RUN npm run build

# Expose the server port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production
ENV PYTHON_PATH=python3

# Run the server
CMD ["node", "dist/server.js"]
