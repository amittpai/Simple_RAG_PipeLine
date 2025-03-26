import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("OPENAI_API_KEY")
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

db_client = chromadb.PersistentClient(path = CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)

# just override if already exists or create a new collection
collection = db_client.get_or_create_collection(
    name='Urology_Research',
    embedding_function = openai_ef
)

# Load the Document
pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
# put it in a list
raw_doc = pdf_loader.load()

# split the text in the pdf
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex = False,
)

# chunk the data
chunks = text_splitter.split_documents(raw_doc)

# prepare the db
documents = []
metadata = []
ids = []


for count, chunk in enumerate(chunks):
    if chunk.page_content.strip(): # Avoid empty chunks
        documents.append(chunk.page_content)
        metadata.append(chunk.metadata)
        ids.append(f"ID{count}_{chunk.metadata.get('page', 0)}") # get unique ids

# # add to the DB
collection.upsert(
    documents = documents,
    metadatas = metadata,
    ids = ids
)
# print(documents)

