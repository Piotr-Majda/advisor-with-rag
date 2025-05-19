# 💼 AI Investment Advisor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

A smart investment advisor combining document analysis with real-time web data.

## 🌟 Features

- 📄 PDF document processing and analysis
- 🔍 Real-time web data integration
- 🧠 GPT-4 powered financial recommendations
- 🌐 English language interface
- 🐳 Docker containerization
- 📚 Source citation for transparency

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Core**: OpenAI (GPT-4, Embeddings)
- **Vector Database**: FAISS
- **Web Search**: Serp API
- **PDF Processing**: PyPDF
- **Containerization**: Docker

## 📋 Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- API keys:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Serp API Key](https://serpapi.com/) [Optional for web context]

## 🚀 Quick Start (Recommended)

Use the provided Makefile to build, run, and manage all services easily:

```bash
# Clone repository
git clone https://github.com/yourusername/investment-advisor-rag.git
cd investment-advisor-rag

# Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys

# Build all services
make build

# Start all services
make run

# Stop all services
make stop

# Clean up containers and volumes
make clean
```

- Access the UI at: http://localhost:8501
- API Gateway: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🚀 Development Mode

For hot-reload and development workflow:

```bash
make dev
```

Or, for a quick start without cleanup:

```bash
make dev-quick
```

## 🧪 Testing & Linting

Run all service tests in parallel:

```bash
make test-all
```

Lint and type-check the codebase:

```bash
make lint
```

## 📂 Code Structure

```
├── Makefile                # Project orchestration commands
├── docker-compose.yml      # Service orchestration
├── services/               # All microservices live here
│   ├── chat-service/
│   ├── document-processing-service/
│   ├── search-service/
│   ├── api-gateway/
│   └── ui-service/
├── shared/                 # Shared libraries/utilities
├── .env.example            # Environment variable template
├── README.md               # This documentation
└── ...
```

## 🔑 Environment Variables

```
OPENAI_API_KEY=your-openai-key
SERP_API_KEY=your-serp-key
```

## 🖥️ Usage

- Upload PDF documents (financial reports, prospectuses) via the UI
- Ask investment-related questions in Polish
- Get AI-powered recommendations combining document analysis, current market data, and financial insights

## 🛠️ Service Development

Each service can be developed and tested independently:

```bash
cd services/<service-name>
pip install -r requirements.txt
# For FastAPI services:
uvicorn app:app --reload --port <port>
# For UI service:
streamlit run app.py
```

## 🏗️ Deployment

The application is cloud-provider independent and can be deployed to AWS, GCP, Azure, or any Kubernetes cluster. See the Deployment section below for details.

## 🧹 Cleaning Up

Remove all containers, volumes, and caches:

```bash
make clean-all
```

## 📖 More

- For advanced commands, see the Makefile.
- For environment-specific runs (dev/prod):
  - `make run-dev` or `make run-prod`
  - `make build-dev` or `make build-prod`

## 📄 License

MIT License - See LICENSE for details

## 🤝 Contributing

Contributions welcome! Please:

- Fork the repository
- Create your feature branch
- Submit a Pull Request

## 🙏 Acknowledgments

- OpenAI for advanced AI models
- Serp API for search capabilities

## Note

Always verify financial advice with certified professionals. AI recommendations should be one of multiple factors in investment decisions.
