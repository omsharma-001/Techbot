from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

app = FastAPI()

class Query(BaseModel):
    query: str

ollama = None
vectorstore = None
qa_chain = None

try:
    ollamas = Ollama(base_url='http://ollama:11434', model="llama3")
    web_loader1 = WebBaseLoader("https://en.wikipedia.org/wiki/Web_development")
    web_loader2 = WebBaseLoader("https://en.wikipedia.org/wiki/Mobile_app_development")
    
    data1 = web_loader1.load()
    data2 = web_loader2.load()
    
    combined_data = data1 + data2

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(combined_data)

    oembed = OllamaEmbeddings(base_url="http://ollama:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
   
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    qa_chain = RetrievalQA.from_chain_type(
        ollamas,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
   
except Exception as e:
    print(f"Error during initialization: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get():
    file_path = "index.html"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    return HTMLResponse(content="<h1>File not found</h1>", status_code=404)

@app.post("/answer")
async def answer_question(query: Query):
    if not ollama or not vectorstore or not qa_chain:
        raise HTTPException(status_code=500, detail="Server not initialized properly")

    try:
        user_query = query.query
        res = qa_chain({"query": user_query})
        return JSONResponse(content={"result": res['result']})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
