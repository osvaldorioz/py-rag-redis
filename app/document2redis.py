import redis
from sentence_transformers import SentenceTransformer

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "El cielo es azul debido a la dispersión de Rayleigh.",
    "Los gatos son animales domésticos muy populares.",
    "Python es un lenguaje de programación versátil."
]
document_keys = ["doc:1", "doc:2", "doc:3"]

# Almacenar embeddings y texto
embeddings = model.encode(documents).tolist()
for key, doc, emb in zip(document_keys, documents, embeddings):
    r.set(key + ":text", doc)
    r.set(key + ":emb", ','.join(map(str, emb)))

# Almacenar embedding de la query
query = "¿Por qué el cielo es azul?"
query_emb = model.encode([query])[0].tolist()
r.set("query:" + query, ','.join(map(str, query_emb)))

print("Datos almacenados en Redis")