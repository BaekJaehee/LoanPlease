FROM node:18.19.0

WORKDIR /app

COPY package.json /app

RUN npm install
COPY . /app

EXPOSE 5173
RUN npm run build

CMD ["npm", "run", "dev", "--host=0.0.0.0"]
