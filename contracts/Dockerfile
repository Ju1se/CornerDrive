FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY hardhat.config.js ./
COPY contracts ./contracts/
COPY scripts ./scripts/

CMD ["npx", "hardhat", "compile"]
