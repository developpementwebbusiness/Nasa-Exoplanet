
FROM node:24.9-alpine3.21

WORKDIR /Website ./

COPY Website/package*.json ./
RUN npm install
COPY Website/ ./

EXPOSE 3000

#Run dev mode with hot reload
CMD ["npm", "run", "dev"]



