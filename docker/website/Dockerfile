FROM node:24.9-alpine3.21

# Set working directory
WORKDIR /Website

# Install pnpm globally
RUN npm install -g pnpm

# Copy only package files first (for caching)
COPY Website/package*.json ./

# Install dependencies using pnpm
RUN pnpm install --frozen-lockfile

# Copy the rest of the source code
COPY Website/ .

# Expose the port your app runs on
EXPOSE 3000

# Run in dev mode with hot reload
CMD ["pnpm", "run", "dev"]
