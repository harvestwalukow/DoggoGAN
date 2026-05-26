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

# Copy frontend (Vite) dependencies and install
COPY infinite-canvas-image-generation/package*.json /app/infinite-canvas-image-generation/
WORKDIR /app/infinite-canvas-image-generation
RUN npm install

# Copy backend (Node) dependencies and install
COPY infinite-canvas-image-generation/server/package*.json /app/server/
WORKDIR /app/server
RUN npm install

# Copy all project files
WORKDIR /app
COPY dog_gan_64.pth /app/
COPY inference.py /app/
COPY train_code.py /app/
COPY infinite-canvas-image-generation /app/infinite-canvas-image-generation
COPY infinite-canvas-image-generation/server /app/server

# Build the frontend
WORKDIR /app/infinite-canvas-image-generation
RUN npm run build

# Build the TypeScript server
WORKDIR /app/server
RUN npm run build

# Copy frontend built files to the server's static directory
RUN cp -r /app/infinite-canvas-image-generation/dist/* /app/server/dist/

# Expose the server port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production
ENV PYTHON_PATH=python3

# Run the server
CMD ["node", "dist/server.js"]

