import os

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

if "OPENAI_API_KEY" not in os.environ:
    raise SystemExit("⚠️  Exporta OPENAI_API_KEY antes de ejecutar.")

docs = TextLoader("sample.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(chunks, emb, collection_name="demo")
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"), chain_type="stuff", retriever=retriever
)
print(qa.run("¿De qué trata el documento?"))
