# app.py
import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from atlassian import Confluence
from langchain.docstore.document import Document
from urllib.parse import unquote

from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader, UnstructuredWordDocumentLoader
)

# ==== Load .env ====
load_dotenv()
FOLDER_PATH = os.getenv("FOLDER_PATH")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_PAGE_IDS = ["65695"]

# ==== Load & Embed ====
documents = []
LOADERS = {
    ".pdf": PyPDFLoader, ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader, ".xls": UnstructuredExcelLoader,
    ".docx": UnstructuredWordDocumentLoader
}
for filename in os.listdir(FOLDER_PATH):
    ext = os.path.splitext(filename)[1].lower()
    if ext in LOADERS:
        try:
            print("File loaded --> "+filename)
            loader = LOADERS[ext](os.path.join(FOLDER_PATH, filename))
            documents.extend(loader.load())
        except Exception: pass

def load_confluence_pages():
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_API_TOKEN
    )
    docs = []
    for page_id in CONFLUENCE_PAGE_IDS:
        try:
            page = confluence.get_page_by_id(page_id, expand='body.storage')
            html_body = page['body']['storage']['value']
            text = html_body.replace('<br/>', '\n')
            docs.append(Document(page_content=text, metadata={"source": f"Page {page_id}"}))
        except Exception: pass
    return docs

documents += load_confluence_pages()
split_docs = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_documents(documents)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=PERSIST_DIRECTORY)
vectorstore.persist()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ==== API ====
app = FastAPI()

# CORS to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/run-rag")
def run_rag(query: str = Query(...)):
    try:
        query = unquote(query)  # Converts %20 into spaces
        print("Received Query:", query)
        result = qa({"query": query})
        return {"answer": result["result"]}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")