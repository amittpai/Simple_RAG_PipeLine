from openai import OpenAI
from dotenv import load_dotenv
import chroma_db
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()
api_key = getenv("OPENAI_API_KEY")
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

db_client = chroma_db.PersistentClient(path = CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)

# just override if already exists or create a new collection
collection = db_client.get_or_create_collection(
    name='Urology_Research',
    embedding_function = openai_ef
)

user_query = input('''How to evaluate the clinical efficacy of percutaneous nephrolithotomy
                    versus retrograde intrarenal surgery''')

results = collection.query(
    query = [user_query],
    n_results = 1,
)

client = OpenAI()

system_prompt = '''
You are a helpful assistant. You are helping a researcher evaluate the clinical efficacy of percutaneous nephrolithotomy versus retrograde intrarenal surgery.
But you need to provide more information to the researcher from the data that has been provided to you.
Do not use external sources or knowledge to answer the questions. 
If you do not know the answer, reply saying "I do not know the answer to that question." 
Do not make up any information.

Here is the data that you have:
''' + str(results['documents'])+'''
'''

# Make call to LLM
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "developer", "content": system_prompt},
    {"role": "user", "content": user_query}
  ]
)

# Test the response
print(response.choices[0].message.content)
