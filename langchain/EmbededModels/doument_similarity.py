import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

load_dotenv()

# Get the directory of the current script to find the files reliably
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_file = os.path.join(script_dir, "documents.txt")
embeddings_file = os.path.join(script_dir, "doc_embeddings.npy")

# Read documents from local file
if not os.path.exists(docs_file):
    print(f"Please create '{docs_file}' with your documents.")
    exit(1)

with open(docs_file, "r") as f:
    documents = [line.strip() for line in f if line.strip()]

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Store or Load embeddings
if os.path.exists(embeddings_file):
    print("Loading existing embeddings from file...")
    doc_embeddings = np.load(embeddings_file)
else:
    print("Generating new embeddings and saving to file...")
    doc_embeddings = embedding.embed_documents(documents)
    np.save(embeddings_file, doc_embeddings)

# Take query as input interactively
while True:
    try:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower().strip() == 'exit':
            print("Exiting...")
            break
            
        if not query.strip():
            continue

        query_embedding = embedding.embed_query(query)

        scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

        print("\n--- Result ---")
        print("similar:", documents[index])
        print("Similarity score:", round(score, 4))
        
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
 

# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query = 'tell me about bumrah'

# doc_embeddings = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

# print(query)
# print(documents[index])
# print("similarity score is:", score)



