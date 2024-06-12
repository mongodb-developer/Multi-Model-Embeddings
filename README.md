# Multi-Model-Embeddings 🦜 ⛓️
Example code to show how to chunk + work with multiple embeddings in the same document.
You can also create the vector indexes in the Atlas UI - but there is code here for that also

#query multiple embedded fields

```
import openai
from transformers import BertTokenizer, BertModel
import torch
from pymongo import MongoClient

# Initialize OpenAI and Hugging Face models
openai.api_key = 'your_openai_api_key'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Connect to MongoDB
client = MongoClient('your_mongo_connection_string')
db = client.your_database_name
collection = db.your_collection_name

# Function to create embeddings using OpenAI
def create_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Function to create embeddings using Hugging Face BERT
def create_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Example query text
query_text = "example sentence for query"

# Create embeddings for the query text
query_embedding_openai = create_openai_embedding(query_text)
query_embedding_bert = create_bert_embedding(query_text)

# Convert embeddings to float32 (if needed)
query_embedding_openai = [float(x) for x in query_embedding_openai]
query_embedding_bert = [float(x) for x in query_embedding_bert]

# Find similar documents based on field3_embedding (OpenAI)
results_field3 = collection.aggregate([
    {
        "$search": {
            "index": "field3_embedding_vector_index",
            "vector": {
                "queryVector": query_embedding_openai,
                "path": "field3_embedding",
                "k": 10  # Number of nearest neighbors to retrieve
            }
        }
    }
])

# Find similar documents based on field15_embedding (BERT)
results_field15 = collection.aggregate([
    {
        "$search": {
            "index": "field15_embedding_vector_index",
            "vector": {
                "queryVector": query_embedding_bert,
                "path": "field15_embedding",
                "k": 10  # Number of nearest neighbors to retrieve
            }
        }
    }
])

# Process the results
print("Similar documents for field3_embedding (OpenAI):")
for result in results_field3:
    print(result)

print("\nSimilar documents for field15_embedding (BERT):")
for result in results_field15:
    print(result)
```
