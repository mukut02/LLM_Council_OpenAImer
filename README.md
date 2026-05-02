# LLM Council

## Run React frontend + Python API

1. Install Python deps:
```bash
pip install -r requirements.txt
```

2. Start backend API:
```bash
uvicorn api_server:app --reload --port 8000
```

3. In a second terminal, start frontend:
```bash
cd frontend
npm install
npm run dev
```

4. Open:
- `http://localhost:5173`
