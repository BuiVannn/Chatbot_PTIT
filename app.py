from flask import Flask, request, jsonify, render_template, g
import pickle
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__, static_url_path='/static')

# Tạo các biến toàn cục để lưu trữ các đối tượng cần thiết
rag_chain = None
model = None

# Hàm khởi tạo dữ liệu
def initialize_data():
    global rag_chain, model
    
    if rag_chain is not None:
        # Đã khởi tạo rồi, không cần khởi tạo lại
        return
    
    # Thiết lập API key cho OpenRouter
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("Thiếu API key! Vui lòng thiết lập biến môi trường OPENROUTER_API_KEY")
    
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    
    # Thiết lập base URL cho OpenRouter
    openai.api_base = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    
    # Tải model embedding
    model_name = "vinai/phobert-base-v2"
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
    
    # Tải documents và embeddings
    with open('ptit_documents.pkl', 'rb') as f:
        documents = pickle.load(f)
    
    with open('ptit_embeddings.pkl', 'rb') as f:
        content_embeddings = pickle.load(f)
    
    # Tạo embeddings wrapper
    class PrecomputedEmbeddings(Embeddings):
        def __init__(self, model, docs, embeddings):
            self.model = model
            self.docs = docs
            self.embeddings = embeddings
            self.doc_map = {doc['content']: idx for idx, doc in enumerate(docs)}
        
        def embed_documents(self, texts):
            results = []
            for text in texts:
                if text in self.doc_map:
                    idx = self.doc_map[text]
                    results.append(self.embeddings[idx].tolist())
                else:
                    results.append(self.model.encode(text).tolist())
            return results
        
        def embed_query(self, text):
            return self.model.encode(text).tolist()
    
    # Khởi tạo embeddings
    embeddings_wrapper = PrecomputedEmbeddings(model, documents, content_embeddings)
    
    # Tạo vector store
    texts = [doc['content'] for doc in documents]
    metadatas = [{"id": doc['id'], "question": doc['question'], "answer": doc['answer']} for doc in documents]
    faiss_db = FAISS.from_texts(texts=texts, embedding=embeddings_wrapper, metadatas=metadatas)
    
    # Tạo retriever
    retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
    
    # Khởi tạo LLM
    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.1-24b-instruct:free",
        temperature=0.3,
        #headers={"HTTP-Referer": "https://ptit.edu.vn", "X-Title": "PTIT Chatbot"}
    )
    
    # Tạo template prompt
    template = """Bạn là một trợ lý AI giúp sinh viên và người quan tâm tìm hiểu về Học viện Công nghệ Bưu chính Viễn thông (PTIT).
    Hãy sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác nhất.
    Nếu không có đủ thông tin để trả lời, hãy thừa nhận điều đó và đề nghị liên hệ trực tiếp với nhà trường.
    
    Thông tin tham khảo:
    {context}
    
    Câu hỏi: {question}
    
    Trả lời bằng tiếng Việt, trình bày rõ ràng, mạch lạc và lịch sự:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Tạo RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Sử dụng before_request để khởi tạo trước khi xử lý bất kỳ request nào
@app.before_request
def before_request():
    initialize_data()

# Trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint để xử lý truy vấn
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "Vui lòng nhập câu hỏi"}), 400
    
    try:
        result = rag_chain.invoke(query_text)
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Khởi tạo dữ liệu trước khi chạy ứng dụng
    initialize_data()
    app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import pickle
# import os
# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import FAISS
# from langchain_core.embeddings import Embeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import openai

# app = Flask(__name__)

# # Khởi tạo mô hình và dữ liệu
# @app.before_first_request
# def initialize():
#     global rag_chain, model
#     OPENROUTER_API_KEY = "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"
#     # Thiết lập API key cho OpenRouter
#     OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your_openrouter_api_key")
#     os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    
#     # Thiết lập base URL cho OpenRouter
#     openai.api_base = "https://openrouter.ai/api/v1"
#     os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    
#     # Tải model embedding
#     model_name = "vinai/phobert-base-v2"
#     try:
#         model = SentenceTransformer(model_name)
#     except Exception:
#         model_name = "all-MiniLM-L6-v2"
#         model = SentenceTransformer(model_name)
    
#     # Tải documents và embeddings
#     with open('ptit_documents.pkl', 'rb') as f:
#         documents = pickle.load(f)
    
#     with open('ptit_embeddings.pkl', 'rb') as f:
#         content_embeddings = pickle.load(f)
    
#     # Tạo embeddings wrapper
#     class PrecomputedEmbeddings(Embeddings):
#         def __init__(self, model, docs, embeddings):
#             self.model = model
#             self.docs = docs
#             self.embeddings = embeddings
#             self.doc_map = {doc['content']: idx for idx, doc in enumerate(docs)}
        
#         def embed_documents(self, texts):
#             results = []
#             for text in texts:
#                 if text in self.doc_map:
#                     idx = self.doc_map[text]
#                     results.append(self.embeddings[idx].tolist())
#                 else:
#                     results.append(self.model.encode(text).tolist())
#             return results
        
#         def embed_query(self, text):
#             return self.model.encode(text).tolist()
    
#     # Khởi tạo embeddings
#     embeddings_wrapper = PrecomputedEmbeddings(model, documents, content_embeddings)
    
#     # Tạo vector store
#     texts = [doc['content'] for doc in documents]
#     metadatas = [{"id": doc['id'], "question": doc['question'], "answer": doc['answer']} for doc in documents]
#     faiss_db = FAISS.from_texts(texts=texts, embedding=embeddings_wrapper, metadatas=metadatas)
    
#     # Tạo retriever
#     retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
    
#     # Khởi tạo LLM
#     llm = ChatOpenAI(
#         model="meta-llama/llama-3.3-8b-instruct:free",
#         temperature=0.3,
#         #headers={"HTTP-Referer": "https://ptit.edu.vn", "X-Title": "PTIT Chatbot"}
#     )
    
#     # Tạo template prompt
#     template = """Bạn là một trợ lý AI giúp sinh viên và người quan tâm tìm hiểu về Học viện Công nghệ Bưu chính Viễn thông (PTIT).
#     Hãy sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác nhất.
#     Nếu không có đủ thông tin để trả lời, hãy thừa nhận điều đó và đề nghị liên hệ trực tiếp với nhà trường.
    
#     Thông tin tham khảo:
#     {context}
    
#     Câu hỏi: {question}
    
#     Trả lời bằng tiếng Việt, trình bày rõ ràng, mạch lạc và lịch sự:
#     """
    
#     prompt = PromptTemplate.from_template(template)
    
#     def format_docs(docs):
#         return "\n\n".join([doc.page_content for doc in docs])
    
#     # Tạo RAG chain
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

# # Trang chủ
# @app.route('/')
# def home():
#     return render_template('index.html')

# # API endpoint để xử lý truy vấn
# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     query_text = data.get('query', '')
    
#     if not query_text:
#         return jsonify({"error": "Vui lòng nhập câu hỏi"}), 400
    
#     try:
#         result = rag_chain.invoke(query_text)
#         return jsonify({"answer": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)