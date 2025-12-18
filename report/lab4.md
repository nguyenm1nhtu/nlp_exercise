# Lab4: Text Classification & Sentiment Analysis

### 1. Các bước triển khai

**Task 1: Scikit-learn TextClassifier**
Cài đặt class TextClassifier với fit / predict / evaluate (LogisticRegression + vector hóa văn bản).

**Task 2: Basic Test Case**
Script test tách train/test, tiền xử lý (tokenize + vectorize), huấn luyện, dự đoán, đánh giá.

**Task 3: Running the Spark Example**
Pipeline Spark ML (Tokenizer → StopWords → HashingTF → IDF → LogisticRegression), chạy và hiểu thành phần.

**Task 4: Model Improvement Experiment**
Thử ít nhất một: preprocessing nâng cao hoặc Word2Vec hoặc mô hình khác (NaiveBayes/GBT/NeuralNet).

### 2. Phân tích kết quả

- Baseline (Spark TF-IDF + LR): độ chính xác & F1 ở mức khá (≈0.7x), train nhanh, ổn định cho dữ liệu văn bản tuyến tính.

- Preprocessing nâng cao (TF-IDF): cải thiện nhẹ Acc/F1 do giảm nhiễu & chọn lọc đặc trưng.

- Word2Vec (trung bình embedding): giảm hiệu quả so với TF-IDF vì mất ngữ cảnh khi chỉ lấy trung bình vector.

- NeuralNet trên TF-IDF: F1 cao nhất trong phạm vi thí nghiệm, tận dụng tương tác phi tuyến giữa đặc trưng.

Kết luận ngắn: Trong phạm vi lab và dữ liệu hiện có, TF-IDF + NeuralNet là lựa chọn khuyến nghị để đạt F1 tốt, còn Word2Vec cần kỹ thuật tổng hợp/ngữ cảnh tốt hơn hoặc embedding pre-trained.

### 3. Khó khăn thực tế và giải pháp

- Chuẩn hóa nhãn: đổi -1 → 0 cho Spark (yêu cầu label không âm).

- Lệch nhãn: kiểm tra phân phối sau lọc; có thể dùng stratify khi chia train/test.

- Bộ nhớ (GBT + đặc trưng lớn): giảm numFeatures ở HashingTF, hoặc giảm n-gram.

- Embedding chất lượng: Word2Vec cần corpora đủ lớn/đa dạng; cân nhắc GloVe/FastText tiền huấn luyện.

- Tái lập: cố định seed, log quy trình tách dữ liệu & tham số.

- Triển khai: cân bằng hiệu quả (F1) với chi phí tính toán/tài nguyên khi mở rộng.

---
