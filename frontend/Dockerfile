FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install --no-fund --no-audit
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]


