import redis
from sentence_transformers import SentenceTransformer

# Conectar a Redis
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documentos de ejemplo
documents = [
    "El cielo es azul debido a la dispersión de Rayleigh.",
    "Los gatos son animales domésticos muy populares.",
    "Python es un lenguaje de programación versátil."
]
document_keys = ["doc:1", "doc:2", "doc:3"]

# Generar y almacenar embeddings y texto de documentos
embeddings = model.encode(documents).tolist()
for key, doc, emb in zip(document_keys, documents, embeddings):
    r.set(key + ":text", doc)  # Almacenar el texto
    r.set(key + ":emb", ','.join(map(str, emb)))  # Almacenar el embedding como cadena CSV

print("Datos de documentos almacenados en Redis")
