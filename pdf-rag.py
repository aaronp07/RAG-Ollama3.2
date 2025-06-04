# 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar document
# 6. Retrieve the similar documents and present them to the user

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/Medical_book.pdf"
model = "llama3.2"

# ==== Start - PDF Ingestion ====

# Local PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Loading done...")
else:
    print("Upload a PDF file")
    
    
# Preview first page
content = data[0].page_content
# print(content[:100])

# ==== End - PDF Ingestion ====

# ==== Start - Extract Text from PDF files and split into small chunks ====

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and Chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Splitting done...")

# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunks: {chunks[0]}")

# ==== End - Extract Text from PDF files and split into small chunks ====

# ==== Start - Add to Vector Database ====

import ollama

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag"
)

print("Adding to vector database is done...")

# ==== End - Add to Vector Database ====

# ==== Start - Retrieval ====

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Set up our model to use
llm = ChatOllama(model=model)

print("Model set done...")

# a simple technique to generate multiple questions from a single
# based on those questions, getting the best of both worlds.

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity seach. Provide these alternative questions separated by newlines.
    Original question: {question}
    """
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

print("Vector databse retriever done...")

# RAG Prompt
template = """Answer the question based ONLY on the following context
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

print("Prompting done...")

# res = chain.invoke(input=("what is the document about?"))
# res = chain.invoke(input=("what are the main point in pregnancy?"))
res = chain.invoke(input=("can you summarrize the alcohol-related neurologic disease?"))

print("Triggering the chain invoke result done...")

print(res)

# ==== End - Retrieval ====