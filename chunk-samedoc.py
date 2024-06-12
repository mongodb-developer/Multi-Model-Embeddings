import pymongo
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Connect to MongoDB
client = pymongo.MongoClient('your_mongo_connection_string')
db = client.your_database_name
collection = db.your_collection_name

# Initialize Embedding Models
openai_embeddings = OpenAIEmbeddings(api_key='your_openai_api_key')
huggingface_embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')

# Function to chunk text and generate embeddings
def process_and_store_text(document_id, text, field_name, embeddings_model, chunk_size=500):
    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings for each chunk
    embeddings = [embeddings_model.get_embedding(chunk) for chunk in chunks]
    
    # Store chunks and their embeddings in MongoDB
    collection.update_one(
        {"_id": document_id},
        {"$set": {f"{field_name}_chunks": chunks, f"{field_name}_embeddings": embeddings}},
        upsert=True
    )

# Example document ID and text
document_id = 'example_document_id'
text_field3 = "Your long text for field 3 goes here..."
text_field15 = "Your long text for field 15 goes here..."

# Process and store text for field 3 using OpenAI embeddings
process_and_store_text(document_id, text_field3, "field3", openai_embeddings)

# Process and store text for field 15 using Hugging Face BERT embeddings
process_and_store_text(document_id, text_field15, "field15", huggingface_embeddings)

print("Chunks and embeddings stored successfully.")
