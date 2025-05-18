# PTIT RAG Chatbot

Chatbot sử dụng công nghệ Tìm kiếm Tăng cường bằng Sinh văn bản (RAG) để trả lời các câu hỏi về Học viện Công nghệ Bưu chính Viễn thông (PTIT). Chatbot này áp dụng các kỹ thuật xử lý ngôn ngữ tự nhiên tiên tiến để cung cấp câu trả lời chính xác cho các truy vấn về chương trình đào tạo, học phí, thông tin tuyển sinh và các chủ đề liên quan khác của PTIT.

## Tính năng

- **Semantic Search**: Sử dụng embeddings để tìm câu trả lời phù hợp nhất với câu hỏi của người dùng
- **Hỗ trợ tiếng việt**: Được tối ưu hóa cho văn bản tiếng Việt với mô hình phobert-base-v2
- **Web Interface**: Giao diện web Gradio dễ sử dụng để tương tác với chatbot
- **RAG Architecture**: Kết hợp phương pháp truy xuất thông tin và sinh văn bản để tạo câu trả lời chính xác

## Cấu trúc Project

chatbot_rag/ 
├── .env                      
├── .env_example              
├── .gitignore                
├── app.py                    
├── chatbot_PTIT_v2.ipynb     
├── datachatbot.json          
├── ptit_documents.pkl        
├── ptit_embeddings.pkl       
├── README.md                 
├── requirements.txt          
├── static/                   
│   └── logo_ptit.png         
└── templates/                
    └── index.html            

## Cài đặt

**1. Clone the repository:**
   ```bash
   git clone <repository-url>
   cd chatbot_rag
   ```

**2. Tạo môi trường ảo:**
```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
```
**3. Cài đặt thư viện cần thiết:**
```bash
    pip install -r requirements.txt
```
**4. Thêm API Key vào file .env:**
```bash
    cp .env_example .env
```
**Sử dụng**
Chạy ứng dụng:

```bash
    flask run
    # or
    python app.py
```
**Sử dụng Chatbot**
- Nhập câu hỏi về PTIT
- Nhấn gửi
- Sử dụng các câu hỏi gợi ý

**Models sử dụng**
- Embedding Model: VINAI's phobert-base-v2 (fallback to all-MiniLM-L6-v2)
- LLM: Meta's Llama-3.3-8b-instruct qua OpenRouter API

