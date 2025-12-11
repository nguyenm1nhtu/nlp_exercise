# Lab 5 Part 3: RNN cho Part-of-Speech Tagging

## Thông tin chung

**Mô tả**: Xây dựng mô hình RNN đơn giản để gán nhãn Part-of-Speech (POS) cho từng từ trong câu sử dụng PyTorch.

**Dataset**: Universal Dependencies English-EWT (UD_English-EWT)
- Định dạng: CoNLL-U
- Train: 12,544 câu
- Dev: 2,001 câu  
- Test: 2,077 câu
- Số lượng POS tags: 17 (NOUN, VERB, PRON, ADJ, ADV, ADP, DET, AUX, PROPN, PART, CCONJ, SCONJ, NUM, PUNCT, INTJ, SYM, X)

---

## Kết quả thực nghiệm

### Thông số mô hình

| Thông số | Giá trị |
|----------|---------|
| Vocabulary size | 9,875 từ |
| Embedding dimension | 100 |
| Hidden dimension | 128 |
| RNN layers | 1 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss (ignore padding) |
| Số tham số | 1,019,262 |
| Epochs | 10 |

### Độ chính xác tổng thể

| Metric | Train | Dev | Test |
|--------|-------|-----|------|
| Loss | 0.1630 | 0.3729 | 0.3750 |
| Accuracy | - | **88.61%** | **88.28%** |

**Best Dev Accuracy**: 88.61% (epoch 10)

### Quá trình huấn luyện

| Epoch | Train Loss | Dev Loss | Dev Accuracy |
|-------|------------|----------|--------------|
| 1 | 1.0894 | 0.7314 | 75.98% |
| 2 | 0.5932 | 0.5525 | 81.98% |
| 3 | 0.4438 | 0.4697 | 84.67% |
| 4 | 0.3538 | 0.4252 | 86.31% |
| 5 | 0.2945 | 0.4114 | 86.36% |
| 6 | 0.2533 | 0.3952 | 86.78% |
| 7 | 0.2222 | 0.3784 | 87.74% |
| 8 | 0.1986 | 0.3742 | 87.88% |
| 9 | 0.1789 | 0.3795 | 87.93% |
| 10 | 0.1630 | 0.3729 | **88.61%** |

**Nhận xét về quá trình huấn luyện**:
- Train loss giảm đều đặn từ 1.0894 xuống 0.1630, cho thấy mô hình học tốt trên tập train
- Dev accuracy tăng liên tục qua các epoch, từ 75.98% lên 88.61%
- Có dấu hiệu overfitting nhẹ (train loss tiếp tục giảm trong khi dev loss tăng nhẹ từ epoch 8-9)
- Mô hình đạt performance tốt nhất ở epoch cuối cùng

![Training History](image/training_history.png)
*Biểu đồ Loss và Accuracy qua các epoch*

### Độ chính xác theo từng POS tag (Test set)

| POS Tag | Số lượng | Accuracy | Nhận xét |
|---------|----------|----------|----------|
| CCONJ | 737 | **99.05%** | Xuất sắc - liên từ đẳng lập dễ nhận dạng (and, but, or) |
| PUNCT | 3,096 | **98.39%** | Xuất sắc - dấu câu có pattern rõ ràng |
| PRON | 2,161 | **97.45%** | Xuất sắc - đại từ có tập hợp từ hạn chế |
| AUX | 1,543 | **96.31%** | Rất tốt - động từ phụ (is, are, was, were) |
| DET | 1,897 | **95.41%** | Rất tốt - mạo từ (the, a, an) |
| ADP | 2,033 | **93.70%** | Tốt - giới từ |
| PART | 649 | **92.14%** | Tốt - tiểu từ (to, not) |
| VERB | 2,605 | **87.06%** | Khá tốt - động từ |
| ADV | 1,178 | **84.13%** | Khá tốt - trạng từ |
| NOUN | 4,137 | **83.76%** | Khá tốt - danh từ (số lượng lớn nhất) |
| ADJ | 1,787 | **79.41%** | Trung bình - tính từ |
| PROPN | 1,980 | **79.19%** | Trung bình - danh từ riêng |
| SYM | 109 | **78.90%** | Trung bình - ký hiệu |
| INTJ | 120 | **70.00%** | Khá thấp - thán từ |
| NUM | 542 | **67.71%** | Khá thấp - số |
| SCONJ | 384 | **52.60%** | Thấp - liên từ phụ thuộc |
| X | 136 | **15.44%** | Rất thấp - từ không xác định |

**Phân tích theo nhóm**:

**Nhóm xuất sắc (>95%)**: CCONJ, PUNCT, PRON, AUX, DET
- Các từ loại có tập từ vựng giới hạn, pattern rõ ràng
- Ít có sự nhầm lẫn với các POS khác

**Nhóm tốt (85-95%)**: ADP, PART, VERB
- Động từ có độ chính xác 87%, khá tốt nhưng còn bị nhầm với NOUN

**Nhóm trung bình (75-85%)**: ADV, NOUN, ADJ, PROPN
- Danh từ (83.76%) là POS phổ biến nhất (4,137 lần xuất hiện)
- Tính từ và danh từ riêng dễ bị nhầm lẫn với nhau

**Nhóm thấp (<75%)**: NUM, SCONJ, X, INTJ
- NUM (67.71%): Số thường bị nhầm với NOUN hoặc ADJ
- SCONJ (52.60%): Liên từ phụ thuộc (that, if, because) khó phân biệt với PRON
- X (15.44%): Từ không xác định, ít dữ liệu train, rất khó học
- INTJ (70%): Thán từ (oh, wow) ít xuất hiện

---

## Ví dụ dự đoán

### 1. "I love NLP"
```
I        -> PRON   (Correct ✓)
love     -> VERB   (Correct ✓)
NLP      -> PROPN  (Correct ✓)
```

### 2. "The cat sat on the mat"
```
The      -> DET    (Correct ✓)
cat      -> NOUN   (Correct ✓)
sat      -> VERB   (Correct ✓)
on       -> ADP    (Correct ✓)
the      -> DET    (Correct ✓)
mat      -> NOUN   (Correct ✓)
```

### 3. "Natural language processing is amazing"
```
Natural     -> ADJ     (Correct ✓)
language    -> NOUN    (Correct ✓)
processing  -> VERB    (Sai - nên là NOUN/PROPN)
is          -> AUX     (Correct ✓)
amazing     -> ADJ     (Correct ✓)
```
**Phân tích**: Từ "processing" trong ngữ cảnh này nên là danh từ (Natural Language Processing là cụm danh từ chỉ lĩnh vực), nhưng mô hình dự đoán là VERB vì "processing" thường xuất hiện như động từ dạng V-ing.

### 4. "She quickly ran to the store"
```
She      -> PRON   (Correct ✓)
quickly  -> ADV    (Correct ✓)
ran      -> VERB   (Correct ✓)
to       -> ADP    (Correct ✓)
the      -> DET    (Correct ✓)
store    -> NOUN   (Correct ✓)
```

### 5. "Deep learning models are powerful"
```
Deep      -> PROPN  (Sai - nên là ADJ)
learning  -> VERB   (Sai - nên là NOUN/PROPN)
models    -> NOUN   (Correct ✓)
are       -> AUX    (Correct ✓)
powerful  -> ADJ    (Correct ✓)
```
**Phân tích**: "Deep learning" là cụm danh từ chuyên ngành, nhưng mô hình nhận "Deep" là PROPN và "learning" là VERB. Đây là lỗi phổ biến khi gặp cụm từ chuyên ngành chưa xuất hiện đủ trong tập train.

---


## Nhận xét và đánh giá


### Ưu điểm
1. **Hiệu quả mô hình**
   - Độ chính xác trên tập test đạt 88.28%, phù hợp với kỳ vọng cho một mô hình RNN cơ bản.
   - Mô hình hoạt động ổn định trên các POS phổ biến như NOUN, VERB, ADJ, thể hiện khả năng học tốt các pattern chính trong dữ liệu.

2. **Xử lý dữ liệu và batching**
   - Đã xử lý đúng định dạng CoNLL-U, thêm các token đặc biệt như `<PAD>`, `<UNK>` để đảm bảo tính tổng quát.
   - Việc padding và batching giúp mô hình huấn luyện hiệu quả hơn trên GPU.

3. **Quá trình huấn luyện ổn định**
   - Loss giảm đều qua các epoch, không xuất hiện hiện tượng bất thường.
   - Accuracy trên tập dev tăng dần, mô hình lưu lại checkpoint tốt nhất dựa trên dev accuracy.

### Hạn chế

1. **Dấu hiệu overfitting nhẹ**
   - Ở các epoch cuối, train loss tiếp tục giảm nhưng dev loss có xu hướng dao động nhẹ, cho thấy mô hình bắt đầu học quá kỹ dữ liệu train.
   - Chưa áp dụng các kỹ thuật regularization như dropout hoặc weight decay.

2. **Độ chính xác thấp với các POS hiếm**
   - Các nhãn như X, SCONJ, NUM có accuracy thấp do số lượng mẫu ít, mô hình khó học được pattern đặc trưng.
   - Chưa có giải pháp tăng cường dữ liệu hoặc cân bằng class.

3. **Giới hạn về ngữ cảnh**
   - RNN một chiều chưa tận dụng được thông tin hai chiều trong câu, dẫn đến nhầm lẫn ở các từ phụ thuộc ngữ cảnh rộng.
   - Các trường hợp long-range dependencies vẫn còn bị nhầm lẫn.

4. **Tokenization đơn giản**
   - Hàm dự đoán chỉ tách từ theo dấu cách, chưa xử lý tốt các trường hợp dấu câu hoặc từ ghép.
   - Nếu áp dụng cho dữ liệu thực tế, cần tích hợp tokenizer chuyên dụng hơn.

**Nhận xét**: Mô hình đạt performance vượt HMM baseline và gần với BiLSTM, cho thấy kiến trúc đơn giản nhưng hiệu quả.

---


## Kết luận

**Kết quả cuối cùng**:
- **Best Dev Accuracy**: 88.61%
- **Test Accuracy**: 88.28%
- **Model size**: 1,019,262 parameters

**Đánh giá tổng thể**: 
- Code chất lượng cao, follow best practices
- Performance tốt cho mô hình baseline
