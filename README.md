# 💼 AI Investment Advisor with RAG

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

A smart investment advisor combining document analysis with real-time web data using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- 📄 PDF document processing and analysis
- 🔍 Real-time web data integration
- 🧠 GPT-4 powered financial recommendations
- 🇵🇱 Polish language interface
- 🐳 Docker containerization
- 📚 Source citation for transparency

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Core**: OpenAI (GPT-4, Embeddings)
- **RAG Framework**: LangChain
- **Vector Database**: FAISS
- **Web Search**: Serp API
- **PDF Processing**: PyPDF
- **Containerization**: Docker

## 📋 Prerequisites

- Docker installed
- API keys:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Serp API Key](https://serpapi.com/) [Optional for web context]

## 🚀 Installation

- Clone repository:
    ```bash
    git clone https://github.com/yourusername/investment-advisor-rag.git
    cd investment-advisor-rag
    ```
- Create .env file:
    ```bash
    cp .env.example .env
    # Edit with your API keys
    ```
- Build Docker image:
    ```bash
    docker build -t investment-advisor .
    ```
- Run container:
    ```bash
    docker run -p 8501:8501 --env-file .env investment-advisor
    ```
- Access the app at http://localhost:8501

## 🚀 Alternative Docker Compose Installation

- Build and start:
  ```bash
  docker-compose up --build
  ```
- Stop and remove:
  ```bash
  docker-compose down
  ```
- For clean rebuild:
  ```bash
  docker-compose down && docker-compose up --build
  ```

## 🖥️ Usage
- Upload PDF documents (financial reports, prospectuses)

- Ask investment-related questions in Polish:

    - "W co powinienem zainwestować 100 000 zł?"

    - "Jakie są perspektywy rynku nieruchomości?"

- Get AI-powered recommendations combining:

    - Document analysis

    - Current market data

    - Financial insights
    
## 📂 Code Structure
  ```
  ├── docker-compose.yml
  ├── app.py            - Main application logic
  ├── Dockerfile        - Container configuration
  ├── requirements.txt  - Python dependencies
  ├── .env.example      - Environment template
  ├── README.md         - This documentation
  └── web_search.py     - web search logic
  ```
## 🔑 Environment Variables
  ```
  OPENAI_API_KEY=your-openai-key
  SERP_API_KEY=your-serp-key
  ```
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

- LangChain community for RAG framework

## Note
Always verify financial advice with certified professionals. AI recommendations should be one of multiple factors in investment decisions.
