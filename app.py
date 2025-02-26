import streamlit as st
import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from web_search import web_search

logging.basicConfig(level=logging.INFO)

load_dotenv()

st.set_page_config(page_title="💬 AI Doradca Inwestycyjny", page_icon="💼")

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 20px;
}

.user-message {
    background-color: #e3f2fd;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 80%;
    float: right;
    clear: both;
}

.assistant-message {
    background-color: #f5f5f5;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 80%;
    float: left;
    clear: both;
}

.clearfix::after {
    content: "";
    clear: both;
    display: table;
}
</style>
""", unsafe_allow_html=True)

VECTOR_DB_PATH = "faiss_index"


def save_vector_store(vector_store):
    """Saves FAISS vector store using LangChain's built-in method"""
    try:
        vector_store.save_local(VECTOR_DB_PATH)
        print("FAISS index saved successfully!")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")


def load_vector_store():
    """Loads FAISS vector store with proper embeddings"""
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                folder_path=VECTOR_DB_PATH,
                embeddings=OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    return None


def process_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Use a context manager for temp file handling
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())

            # Verify file was written correctly
            if os.path.getsize(tmp_file.name) == 0:
                st.error(f"Empty file detected: {uploaded_file.name}")
                continue

            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load()
            documents.extend(pages)
        except Exception as e:
            st.error(f"Błąd przetwarzania pliku {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def generate_response(question, vector_store):
    logging.info(f"Generating AI response for question: {question}")
    doc_context = ""
    if vector_store:
        docs = vector_store.similarity_search(question, k=3)
        doc_context = "\n\n".join([doc.page_content for doc in docs])
    logging.info(f"Document context: \n{doc_context}")

    web_context = web_search(question)

    logging.info(f"Web context: \n{web_context}")

    template = """
    Jesteś ekspertem finansowym. Odpowiadaj w języku polskim. Użyj następujących informacji:

    [DOKUMENTY UŻYTKOWNIKA]:
    {doc_context}

    [AKTUALNE DANE Z INTERNETU]:
    {web_context}

    Pytanie: {question}
    Odpowiedź:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["doc_context", "web_context", "question"]
    )

    llm = ChatOpenAI(temperature=0.3, model="gpt-4-0125-preview")
    chain = prompt | llm

    return chain.invoke({
        "doc_context": doc_context,
        "web_context": web_context,
        "question": question
    })


def main():
    st.title("💬 AI Doradca Inwestycyjny")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()

    with st.sidebar:
        st.header("📁 Zarządzanie dokumentami")
        uploaded_files = st.file_uploader(
            "Prześlij pliki PDF",
            type="pdf",
            accept_multiple_files=True,
            help="Maksymalnie 10 plików PDF"
        )
        new_files = []
        if uploaded_files:
            new_files = [file for file in uploaded_files if
                         file.name not in st.session_state.get("processed_files", [])]

        if new_files:
            try:
                with st.spinner("Przetwarzanie dokumentów..."):
                    for file in uploaded_files:
                        if file.size > 30 * 1024 * 1024:  # 30MB limit
                            raise ValueError(f"Plik {file.name} jest zbyt duży (max 30MB)")
                    texts = process_pdfs(uploaded_files)
                    embeddings = OpenAIEmbeddings()

                    # Create new FAISS index for new documents
                    new_vector_store = FAISS.from_documents(texts, embeddings)

                    if st.session_state.vector_store:
                        st.session_state.vector_store.merge_from(new_vector_store)
                    else:
                        st.session_state.vector_store = new_vector_store

                    save_vector_store(st.session_state.vector_store)

                    # Store processed filenames
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []
                    st.session_state.processed_files.extend([file.name for file in new_files])

                    st.success("Dokumenty przetworzone!")
            except Exception as e:
                st.error(f"Błąd przetwarzania: {str(e)}")

    if st.session_state.messages:
        chat_container = st.container()
        with chat_container:
            st.markdown('<div>', unsafe_allow_html=True)

            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}</div><div class="clearfix"></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="assistant-message">{message["content"]}</div><div class="clearfix"></div>',
                        unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Zadaj pytanie dotyczące inwestycji..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analizowanie..."):
            response = generate_response(prompt, st.session_state.vector_store)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content
            })

        st.rerun()


if __name__ == "__main__":
    main()