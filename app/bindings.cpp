#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sw/redis++/redis++.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace sw::redis;

class RAG {
public:
    RAG(const std::string& redis_host = "127.0.0.1", int redis_port = 6379);
    std::string generate_response(const std::string& query, 
                                const std::vector<std::string>& document_keys);
    
private:
    Redis redis;
    
    std::vector<std::pair<std::string, std::vector<float>>> get_docs_and_embeddings(
        const std::vector<std::string>& keys);
    std::vector<std::string> retrieve_relevant_docs(
        const std::string& query,
        const std::vector<std::string>& document_keys);
    std::vector<float> get_query_embedding(const std::string& query);
};

RAG::RAG(const std::string& redis_host, int redis_port) 
    : redis("tcp://" + redis_host + ":" + std::to_string(redis_port)) {
    try {
        auto pong = redis.ping();
        std::cout << "Conexión a Redis exitosa: " << pong << std::endl;
    } catch (const Error& e) {
        throw std::runtime_error("Error conectando a Redis: " + std::string(e.what()));
    }
}

std::vector<float> RAG::get_query_embedding(const std::string& query) {
    OptionalString emb_str = redis.get("query:" + query);
    if (!emb_str) {
        throw std::runtime_error("Embedding de la query no encontrado en Redis");
    }
    
    std::vector<float> embedding;
    std::stringstream ss(*emb_str);
    std::string value;
    while (std::getline(ss, value, ',')) {
        embedding.push_back(std::stof(value));
    }
    return embedding;
}

std::vector<std::pair<std::string, std::vector<float>>> RAG::get_docs_and_embeddings(
    const std::vector<std::string>& keys) {
    std::vector<std::pair<std::string, std::vector<float>>> docs_and_embs;
    
    for (const auto& key : keys) {
        // Obtener el embedding
        OptionalString emb_str = redis.get(key + ":emb");
        if (!emb_str) {
            throw std::runtime_error("Embedding no encontrado para la clave: " + key + ":emb");
        }
        
        std::vector<float> embedding;
        std::stringstream ss_emb(*emb_str);
        std::string value;
        while (std::getline(ss_emb, value, ',')) {
            embedding.push_back(std::stof(value));
        }
        
        // Obtener el texto del documento
        OptionalString text = redis.get(key + ":text");
        if (!text) {
            throw std::runtime_error("Texto no encontrado para la clave: " + key + ":text");
        }
        
        docs_and_embs.push_back({*text, embedding});
    }
    
    return docs_and_embs;
}

std::vector<std::string> RAG::retrieve_relevant_docs(
    const std::string& query,
    const std::vector<std::string>& document_keys) {
    auto query_embedding = get_query_embedding(query);
    auto docs_and_embs = get_docs_and_embeddings(document_keys);
    
    std::vector<std::pair<float, int>> similarities;
    for (size_t i = 0; i < docs_and_embs.size(); i++) {
        float dot_product = 0.0f, norm_query = 0.0f, norm_doc = 0.0f;
        for (size_t j = 0; j < query_embedding.size(); j++) {
            dot_product += query_embedding[j] * docs_and_embs[i].second[j];
            norm_query += query_embedding[j] * query_embedding[j];
            norm_doc += docs_and_embs[i].second[j] * docs_and_embs[i].second[j];
        }
        float similarity = dot_product / (std::sqrt(norm_query) * std::sqrt(norm_doc));
        similarities.push_back({similarity, i});
    }
    
    std::sort(similarities.begin(), similarities.end(), 
             [](auto& a, auto& b) { return a.first > b.first; });
    
    std::vector<std::string> relevant_docs;
    relevant_docs.push_back(docs_and_embs[similarities[0].second].first); // Devolver el texto
    return relevant_docs;
}

std::string RAG::generate_response(const std::string& query, 
                                 const std::vector<std::string>& document_keys) {
    auto relevant_docs = retrieve_relevant_docs(query, document_keys);
    std::string prompt = "Query: " + query + "\nRelevant info: " + 
                        relevant_docs[0] + "\nResponse: ";
    return prompt + "Basado en la información en Redis, la respuesta es derivada de: " + 
           relevant_docs[0];
}

PYBIND11_MODULE(rag_module, m) {
    py::class_<RAG>(m, "RAG")
        .def(py::init<const std::string&, int>(), 
             py::arg("redis_host") = "127.0.0.1", 
             py::arg("redis_port") = 6379)
        .def("generate_response", &RAG::generate_response);
}