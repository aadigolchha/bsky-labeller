# Use Node as base
FROM node:20-slim

# Install Python & Build Tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Node Deps
COPY package*.json ./
RUN npm install

# 2. Install Python Deps
COPY logic/requirements.txt ./logic/
# --break-system-packages is required in modern Debian/Docker environments
RUN pip3 install -r logic/requirements.txt --break-system-packages

# 3. Copy App Code
COPY . .

# 4. Build TypeScript
RUN npm run build

# 5. Setup Volume & Permissions
RUN mkdir -p /data
ENV DATABASE_URL="file:/data/labeler.db"

# 6. Expose Port (Fly.io expects 8080)
EXPOSE 8080

# 7. Start!
CMD ["./start.sh"]