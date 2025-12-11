# Lab 3: Word Embeddings

## Mục tiêu

Hoàn thành đủ 5 nhóm tác vụ (pretrained, sentence embedding, Word2Vec nhỏ, Word2Vec Spark, trực quan hóa)

## Cấu trúc

- `test/lab4_test.py` → Pretrained (GloVe) : vector, similarity, most-similar
- `src/representations/word_embedder.py` → Hàm nhúng văn bản bằng trung bình vector từ
- `test/lab4_embedding_training_demo.py` → Huấn luyện Word2Vec nhỏ (Gensim) + save/load
- `test/lab4_spark_word2vec_demo.py` → Huấn luyện Word2Vec lớn (Spark MLlib)
- `results/`

## Cài đặt dependencies

```bash
pip install gensim numpy matplotlib scikit-learn pandas
pip install pyspark
```

### 1. Pre-trained Model Test

```bash
cd Lab3
python test/lab4_test.py
```

**Output**: `results/lab4_test_output.txt`

### 2. Custom Word2Vec Training

```bash
python test/lab4_embedding_training_demo.py
```

**Output**: `results/lab4_training_demo_output.txt`

### 3. Spark MLlib Training

```bash
python test/lab4_spark_word2vec_demo.py
```

**Output**: `results/lab4_spark_word2vec_output.txt`

### 4. Visualization

```bash
jupyter notebook Lab3.ipynb
```

## kết quả chi tiết

### 1 Pre-trained Model (GloVe) Analysis

#### Kết quả

- **Model**: GloVe Wiki Gigaword 50D với 400,000 từ
- **Vector quality**: Vectors có giá trị thực tế, không phải zero vectors
- **Similarity scores**:
  - king-queen: 0.78
  - king-man: 0.53

#### Phân tích từ đồng nghĩa cho "computer"

```
1. computers (0.917)  - Dạng số nhiều, hoàn hảo
2. software (0.881)   - Khái niệm liên quan, hợp lý
3. technology (0.853) - Phạm vi rộng hơn, hợp lý
4. electronic (0.813) - Mối quan hệ phần cứng
5. internet (0.806)   - Bối cảnh sử dụng
```

### 2 Custom Word2Vec Training Analysis

#### Kết quả

- **Dataset**: 13,572 sentences từ UD English-EWT
- **Training time**: 1.58 seconds
- **Vocabulary**: 3,772 words
- **Vector dimensions**: 50D

#### Chất lượng

```
Similarities:
- the-man: 0.575
- man-woman: 0.820
```

**Phân tích**:

- Model tự train đạt similarity scores hợp lý
- Mối quan hệ gender (man-woman: 0.820) được học tốt
- Limited vocabulary do dataset nhỏ, nhưng chất lượng ở mức chấp nhận được

### 3 Spark MLlib Large Dataset Analysis

#### Kết quả

- **Dataset**: 29,971 documents từ C4
- **Thời gian training**: 5.85 phút
- **Vocabulary**: 78,930 từ (rất lớn!)
- **Chiều vector**: 100D

**Phân tích**:

- Vocabulary 20x lớn hơn custom model
- Mối quan hệ ngữ nghĩa rõ ràng
- Training trên dataset lớn cho kết quả robust hơn

### 4 Visualization Analysis

#### Phương pháp

- **Giảm chiều**: PCA từ 100D → 2D
- **Phương sai giải thích**: 10-14%
- **Trực quan hóa**: Scatter plot với vector arrows từ gốc tọa độ

**Kết quả**:

1. **Phân cụm không gian**: Các từ có liên quan về mặt ngữ nghĩa có xu hướng gần nhau trong không gian 2D
2. **Mối quan hệ giới tính**: "king" và "queen" có khoảng cách hợp lý, thể hiện khái niệm tương tự nhưng khác giới tính
3. **Mối quan hệ cấp bậc**: "prince", "duke" cluster gần "king", thể hiện thứ bậc hoàng gia
4. **Mối quan hệ bối cảnh**: "kingdom", "castle" gần nhau, thể hiện bối cảnh lĩnh vực

### 3.5 So sánh Models

| Aspect                | Pre-trained GloVe | Custom Word2Vec  | Spark MLlib       |
| --------------------- | ----------------- | ---------------- | ----------------- |
| **Vocabulary**        | 400,000           | 3,772            | 78,930            |
| **Training data**     | Massive web data  | Small EWT corpus | Medium C4 dataset |
| **Quality**           | Excellent         | Good             | Very Good         |
| **Similarity scores** | Very high (0.9+)  | Moderate (0.8)   | High (0.8+)       |
| **Semantic coverage** | Comprehensive     | Limited          | Good              |
| **Training time**     | N/A               | Seconds          | Minutes           |

**Kết luận**:

- Pre-trained model có chất lượng tốt nhất do data training lớn
- Custom model với limited data vẫn học được các mối quan hệ ngữ nghĩa cơ bản
- Spark model cân bằng tốt giữa chất lượng và hiệu suất

## Khó khăn và Giải pháp

**Khó khăn**:

- File C4 dataset lớn (30K documents) gây tràn bộ nhớ khi xử lý với sample_fraction=1.0
- Spark tasks thất bại do không đủ bộ nhớ
- PCA trên 400K vectors mất thời gian
- Matplotlib hiển thị chậm với các biểu đồ scatter lớn

**Giải pháp**:

- Tối ưu Spark configuration với adaptive execution
- Sử dụng `.cache()` cho việc xử lý DataFrames
- Xử lý lỗi robust để cleanup các tài nguyên hệ thống
- Subset vectors cho visualization thay vì full vocabulary
- Tối ưu plotting parameters (alpha, point size)
- Progressive visualization approach

---
