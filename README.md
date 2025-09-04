# BIAN Model Generator Service

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend to generate data models based on the Banking Industry Architecture Network (BIAN) knowledge base.

## Features
- Uses Milvus vector store with documents from `bian-research/out_kb`.
- Embedding requests are sent to the service configured by `EMBEDDINGS_URL`.
- LLM defined by environment variables (default `qwen2:72b`).
- Generates JSON, PlantUML, and SQL artifacts for a requested banking product.

## Development

### Backend
```bash
cd backend
pip install .
```

### Frontend
```bash
cd frontend
npm install
```

### Run
Use Docker Compose to start both backend and frontend:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8123/app`.
