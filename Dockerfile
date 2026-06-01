FROM node:22-bookworm-slim AS frontend

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY index.html vite.config.js eslint.config.js ./
COPY src ./src
RUN npm run build

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY sample-images ./sample-images
COPY README.md ./
COPY --from=frontend /app/dist ./dist

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
