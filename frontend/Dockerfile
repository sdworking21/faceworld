FROM node:18-alpine
WORKDIR /app
COPY ./frontend/package*.json ./frontend/
RUN npm install --prefix ./frontend
COPY frontend ./frontend
WORKDIR /app/frontend
EXPOSE 3000
CMD ["npm", "run", "dev"]
