import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter

def main():
    pinecone.init(api_key="00422970-5a38-48cf-876c-070a3f02f8fe", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    text = open("documentos/economia.txt", "r", encoding= "utf8")
    #print(text.read())
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
    text = open("documentos/ingenieria-electronica.txt", "r", encoding="utf8")
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
    text = open("documentos/ingenieria-civil.txt",  "r", encoding= "utf8")
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
    text = open("documentos/ingenieria-sistemas.txt",  "r", encoding= "utf8")
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
    text = open("documentos/ingenieria-electrica.txt",  "r", encoding= "utf8")
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
    text = open("documentos/ingenieria-industrial.txt",  "r", encoding= "utf8")
    data = Pinecone.from_texts(texts=[text.read()], embedding=embeddings, index_name='taller')
def open(file):
    loader = TextLoader(file, encoding="utf8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.split_documents(documents)
    return docs[0].page_content
def buscar():
    pinecone.init(api_key="00422970-5a38-48cf-876c-070a3f02f8fe", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index("taller", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="refine", retriever=docsearch.as_retriever(search_type="similarity"))
    query = "Cuantos años de acreditación tiene ingeniería de industrial?"
    docs = docsearch.similarity_search(query)
    print(qa.run(query))
buscar()
if __name__ == "__main__":
    main()