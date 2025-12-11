# Lab 3: Word Embeddings

## Phần 1: Trực quan hóa và phân tích embedding

### 1. Source code, dữ liệu, kết quả sử dụng
- [Lab3/Lab3.ipynb](../Lab3/Lab3.ipynb): Notebook chính trực quan hóa embedding
- [Lab3/data/glove.6B/](../Lab3/data/glove.6B/): Pre-trained GloVe vectors
- [Lab3/images/](../Lab3/images/): Hình ảnh trực quan hóa

### 1.1. Dataset sử dụng

**Tên dataset**: GloVe Pre-trained Word Vectors

**Mô tả**: Pre-trained word embeddings được huấn luyện trên 6 tỷ tokens từ Wikipedia 2014 + Gigaword 5.

**Cấu trúc dữ liệu**:
- **Format**: Text file, mỗi dòng gồm: `word` + `vector values` (space-separated)
- **Số lượng**: 400,000 từ vựng
- **Kích thước vector**: 50d, 100d, 200d, 300d (sử dụng 100d)
- **Kiểu dữ liệu**: String (word) + Float values (vector)

**Nguồn**: https://nlp.stanford.edu/projects/glove/

**File sử dụng**: `glove.6B.100d.txt` (~331 MB)

**Lưu ý**: Dataset không được commit lên GitHub. Xem [data/README.md](../data/README.md) để tải.

### 2. Các bước thực hiện
1. Load pre-trained GloVe vectors bằng Gensim
2. Giảm chiều vector từ 100D xuống 2D bằng PCA
3. Trực quan hóa các từ trong không gian 2D bằng matplotlib
4. Tìm kiếm Top K từ tương đồng với một từ bất kỳ (ví dụ: "king")
5. Hiển thị kết quả và phân tích cụm từ, độ tương đồng

### 3. Hướng dẫn chạy code
- Mở notebook `Lab3.ipynb` bằng Jupyter hoặc Colab
- Chạy tuần tự các cell để hiển thị kết quả, hình ảnh trực quan hóa
- Đảm bảo file GloVe vectors đã được giải nén vào đúng thư mục

### 4. Hình ảnh trực quan hóa embedding
#### 4.1 PCA 2D từ GloVe
![GloVe PCA 2D](../Lab3/images/glove_pca_2d.png)
*Hình 1: Trực quan hóa PCA 2D của embedding GloVe*

#### 4.2 Word Representation using PCA
![Word Representation using PCA](../Lab3/images/word_representation_using_pca.png)
*Hình 2: Biểu diễn các từ bằng PCA, vector arrows từ gốc tọa độ*


### 5. Nhận xét về độ tương đồng và các từ đồng nghĩa
- Các từ đồng nghĩa/tương đồng tìm được từ model pre-trained GloVe rất hợp lý, ví dụ "computer" → computers, software, technology
- Độ tương đồng cosine cao cho thấy model học tốt các mối quan hệ ngữ nghĩa

### 6. Phân tích biểu đồ trực quan hóa
- Các từ liên quan được nhóm lại gần nhau, ví dụ cụm hoàng gia (king, queen, prince, kingdom)
- Một số cụm thú vị: công nghệ (computer, software, technology), địa lý (country, city, state)
- Giải thích: GloVe học từ thống kê đồng xuất hiện, PCA bảo toàn khoảng cách tương đối

### 7. So sánh model pre-trained và model tự huấn luyện
- Pre-trained GloVe có chất lượng tốt nhất do data lớn, similarity scores cao
- Model tự huấn luyện (Word2Vec) có vocabulary nhỏ hơn, chất lượng vừa phải nhưng vẫn học được các mối quan hệ cơ bản
- Spark MLlib training trên dataset lớn cho kết quả robust hơn, vocabulary lớn

### 8. Khó khăn và giải pháp
- Xử lý file GloVe lớn, cần đủ RAM
- PCA trên nhiều vector tốn thời gian, cần subset cho visualization
- Khác biệt format vector giữa Gensim và Spark, cần chuẩn hóa
- Đã giải quyết bằng cách subset, tối ưu plotting, dùng wrapper class cho embedding

### 9. Nguồn tham khảo
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)