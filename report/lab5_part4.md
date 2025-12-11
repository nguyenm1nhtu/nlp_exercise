# Lab 5 Part 4: RNN for Named Entity Recognition

## Mục tiêu

Trong bài thực hành này, mô hình RNN được xây dựng để nhận dạng các thực thể tên (Named Entity Recognition - NER) từ văn bản. Bộ dữ liệu CoNLL-2003 được sử dụng với hệ thống gán nhãn IOB (Inside, Outside, Beginning) để phân loại 4 loại thực thể: người (PER), địa điểm (LOC), tổ chức (ORG) và khác (MISC).

## Cấu hình thực nghiệm

### Dữ liệu

- **Bộ dữ liệu**: CoNLL-2003
- **Training**: 14,041 câu
- **Validation**: 3,250 câu
- **Test**: 3,453 câu
- **Từ vựng**: 23,625 từ duy nhất
- **Số nhãn NER**: 9 nhãn (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC)

### Mô hình

**Kiến trúc**: SimpleRNN với 3 lớp chính

- **Embedding layer**: 100 chiều
- **RNN layer**: 128 hidden units
- **Linear layer**: ánh xạ sang 9 nhãn NER
- **Tổng số tham số**: 2,393,101

### Huấn luyện

- **Optimizer**: Adam với learning rate 0.001
- **Loss function**: CrossEntropyLoss (ignore_index=-1 cho padding)
- **Batch size**: 32
- **Số epochs**: 5
- **Early stopping**: Lưu mô hình tốt nhất dựa trên validation accuracy

## Kết quả

### Độ chính xác theo epoch

| Epoch | Train Loss | Validation Accuracy |
|-------|------------|---------------------|
| 1     | 0.6282     | 87.28%              |
| 2     | 0.3629     | 90.34%              |
| 3     | 0.2520     | 92.01%              |
| 4     | 0.1833     | 92.25%              |
| 5     | 0.1352     | 93.11%              |

### Hiệu suất trên tập test

- **Test Accuracy**: 90.41%
- **Best Validation Accuracy**: 93.11%

### Độ chính xác theo từng nhãn

| Nhãn   | Số lượng | Accuracy  |
|--------|----------|-----------|  
| O      | 38,323   | 95.51%    |
| B-PER  | 1,617    | 74.15%    |
| I-PER  | 1,156    | 83.82%    |
| B-LOC  | 1,668    | 69.96%    |
| I-LOC  | 257      | 57.59%    |
| B-ORG  | 1,661    | 52.50%    |
| I-ORG  | 835      | 60.60%    |
| B-MISC | 702      | 57.83%    |
| I-MISC | 216      | 51.85%    |

### Kết quả entity-level

Đánh giá theo từng loại thực thể (precision, recall, F1-score):

| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|  
| PER    | 43.21%    | 67.53% | 52.70%   | 1,617   |
| LOC    | 81.50%    | 66.55% | 73.27%   | 1,668   |
| ORG    | 62.08%    | 48.40% | 54.40%   | 1,661   |
| MISC   | 63.65%    | 52.14% | 57.32%   | 702     |

**Overall F1-Score**: 59.85%

## Phân tích kết quả

### Token-level accuracy

Mô hình đạt accuracy 90.41% khi đánh giá theo từng token. Nhãn "O" (không phải thực thể) có độ chính xác rất cao (95.51%), cho thấy mô hình phân biệt rất tốt giữa token thuộc và không thuộc thực thể.

Điểm đáng chú ý là sự khác biệt giữa các nhãn:
- I-PER đạt accuracy cao (83.82%), tốt hơn B-PER (74.15%)
- B-LOC và B-PER đạt khoảng 70-74%, khá tốt cho việc nhận dạng bắt đầu thực thể
- B-ORG và I-MISC có accuracy thấp nhất (~52%), cho thấy khó khăn trong việc phân loại tổ chức và thực thể khác

### Entity-level F1-score

Khi đánh giá theo toàn bộ thực thể (không phải từng token), Overall F1 là 59.85%. Khoảng cách giữa token-level accuracy (90.41%) và entity-level F1 (59.85%) cho thấy mô hình còn gặp khó khăn trong việc nhận dạng đầy đủ các thực thể có nhiều token.

**Phân tích theo loại thực thể**:

- **LOC (Location)**: Đạt F1 cao nhất (73.27%) với precision 81.50% và recall 66.55%. Precision cao cho thấy mô hình nhận dạng địa điểm khá chính xác, tuy nhiên recall thấp hơn cho thấy còn bỏ sót một số địa điểm.

- **MISC (Miscellaneous)**: F1 là 57.32% với precision 63.65% và recall 52.14%. Loại này chứa các thực thể đa dạng, nhưng precision tốt cho thấy mô hình học được một số pattern.

- **ORG (Organization)**: F1 là 54.40% với precision 62.08% và recall 48.40%. Tên tổ chức có cấu trúc phức tạp, recall thấp cho thấy mô hình bỏ sót nhiều tổ chức.

- **PER (Person)**: F1 là 52.70% với precision 43.21% và recall 67.53%. Pattern đặc biệt - recall cao nghĩa là mô hình nhận dạng được nhiều tên người, nhưng precision thấp cho thấy có nhiều false positives (dự đoán nhầm tokens khác là tên người).

## Ví dụ dự đoán

### Các ví dụ thực tế

#### Ví dụ 1: Thực thể Việt Nam
**Input**: "VNU University is located in Hanoi"

| Token      | Predicted Tag |
|------------|---------------|
| VNU        | B-ORG         |
| University | I-ORG         |
| Hanoi      | B-PER         |

Mô hình nhận dạng đúng "VNU University" là tổ chức  
Nhầm lẫn "Hanoi" thành tên người (B-PER) thay vì địa điểm

---

**Input**: "Nguyen Huu Thang works at FPT Corporation in Ho Chi Minh City"

| Token       | Predicted Tag |
|-------------|---------------|
| Nguyen      | B-PER         |
| Huu         | I-PER         |
| Thang       | I-PER         |
| Corporation | I-ORG         |
| Chi         | I-PER         |
| City        | I-ORG         |

Nhận dạng đúng "Nguyen Huu Thang" là người  
Bỏ lỡ "FPT" (chỉ nhận "Corporation")  
Phân đoạn sai "Ho Chi Minh City"

#### Ví dụ 2: Công ty công nghệ
**Input**: "Apple Inc. was founded by Steve Jobs in California"

| Token      | Predicted Tag |
|------------|---------------|
| Apple      | B-ORG         |
| Inc.       | I-ORG         |
| Steve      | B-PER         |
| Jobs       | I-PER         |
| California | B-LOC         |

Nhận dạng hoàn hảo cả 3 thực thể: công ty, người, địa điểm

---

**Input**: "Microsoft headquarters is in Redmond Washington"

| Token      | Predicted Tag |
|------------|---------------|
| Microsoft  | B-ORG         |
| Redmond    | B-LOC         |
| Washington | B-LOC         |

Nhận dạng đúng công ty và 2 địa điểm  
Tuy nhiên "Redmond Washington" nên là một thực thể liên tục

#### Ví dụ 3: Trường hợp nhầm lẫn
**Input**: "The Amazon river flows through Brazil"

| Token  | Predicted Tag |
|--------|---------------|
| Amazon | B-ORG         |
| Brazil | B-LOC         |

Nhầm "Amazon river" (địa điểm) thành tổ chức (công ty Amazon)  
Nhận dạng đúng "Brazil" là địa điểm

---

**Input**: "Tesla and SpaceX are both led by Elon Musk"

| Token  | Predicted Tag |
|--------|---------------|
| Tesla  | B-PER         |
| SpaceX | B-PER         |
| Elon   | B-PER         |
| Musk   | I-PER         |

Nhầm "Tesla" và "SpaceX" (công ty) thành tên người  
Nhận dạng đúng "Elon Musk" là người

#### Ví dụ 4: Từ tập test (đúng hoàn toàn)
**Sentence**: "Wasim Akram b Harris 4"

| Token  | True Tag | Predicted Tag | Status |
|--------|----------|---------------|--------|
| Wasim  | B-PER    | B-PER         | ✓      |
| Akram  | I-PER    | I-PER         | ✓      |
| Harris | B-PER    | B-PER         | ✓      |

Nhận dạng chính xác 2 tên người

#### Ví dụ 5: Trường hợp phức tạp (nhiều lỗi)
**Sentence**: "New York Commodities Desk , 212-859-1640"

| Token        | True Tag | Predicted Tag | Status |
|--------------|----------|---------------|--------|
| New          | B-ORG    | B-LOC         | ✗      |
| York         | I-ORG    | I-LOC         | ✗      |
| Commodities  | I-ORG    | O             | ✗      |
| 212-859-1640 | O        | B-PER         | ✗      |

Nhầm "New York Commodities Desk" (tổ chức) thành địa điểm và bỏ sót phần cuối  
Số điện thoại bị nhận nhầm là tên người

### Nhận xét từ các ví dụ

**Điểm mạnh**:
- Nhận dạng tốt các thực thể phổ biến: Steve Jobs, California, Microsoft
- Phân đoạn chính xác các tên người nhiều từ: "Nguyen Huu Thang", "Elon Musk"
- Precision cao với các công ty công nghệ nổi tiếng: Apple Inc., Google

**Hạn chế**:
- **Nhầm lẫn đa nghĩa**: Amazon (sông vs công ty), Tesla (người vs công ty)
- **Địa điểm ít gặp**: Hanoi, Paris bị nhận nhầm khi đứng đơn lẻ
- **Thực thể nhiều từ**: "Ho Chi Minh City", "New York Commodities" bị phân đoạn sai
- **Pattern không chuẩn**: Số điện thoại, các từ viết tắt gây nhầm lẫn

## Đánh giá

### Điểm mạnh

**Token classification accuracy**: Mô hình đạt 90.41% accuracy trên tập test, cho thấy khả năng phân loại tốt ở mức token. Accuracy tăng đều qua các epoch từ 87.28% lên 93.11% trên validation set, thể hiện quá trình học ổn định không có dấu hiệu overfitting.

**Nhận dạng địa điểm**: LOC đạt F1-score cao nhất (73.27%) với precision rất tốt (81.50%). Điều này cho thấy kiến trúc RNN đơn giản đã nắm bắt được pattern của địa điểm một cách hiệu quả.

**Phân biệt O và thực thể**: Nhãn "O" đạt 95.51% accuracy, mô hình phân biệt rất tốt giữa các token thuộc và không thuộc thực thể. Đây là nền tảng quan trọng cho bài toán NER.

### Hạn chế

**Entity-level F1 chưa cao**: Overall F1 đạt 59.85%, vẫn thấp hơn nhiều so với token-level accuracy. Nguyên nhân chính là mô hình gặp khó khăn trong việc nhận dạng đầy đủ các thực thể nhiều token.

**Precision thấp với PER**: Tên người có precision chỉ 43.21% mặc dù recall khá cao (67.53%). Điều này cho thấy mô hình có xu hướng "over-predict" - dự đoán quá nhiều tokens là tên người, dẫn đến nhiều false positives.

**Nhầm lẫn giữa các loại**: Từ các ví dụ thực tế, mô hình hay nhầm lẫn giữa các loại thực thể (ví dụ: ORG thành LOC, số điện thoại thành PER). Điều này do thiếu context đầy đủ từ kiến trúc RNN đơn giản.

### So sánh với baseline

CoNLL-2003 là benchmark tiêu chuẩn cho NER. Các mô hình state-of-the-art đạt F1-score trên 90%, trong khi mô hình SimpleRNN của chúng ta đạt 59.85%. Đây là kết quả khá tốt cho một mô hình RNN cơ bản không sử dụng:
- Pre-trained embeddings (Word2Vec, GloVe, BERT)
- Bi-directional RNN
- Conditional Random Fields (CRF) layer
- Character-level embeddings

### Hướng cải thiện

Để nâng cao hiệu suất, có thể:
1. Sử dụng LSTM/GRU thay vì RNN đơn giản để giữ thông tin dài hạn tốt hơn
2. Thêm bi-directional RNN để nắm bắt context từ cả hai hướng
3. Sử dụng pre-trained embeddings như GloVe hoặc Word2Vec
4. Thêm CRF layer để đảm bảo tính nhất quán của chuỗi nhãn (tránh I-PER sau O)
5. Tăng số epochs và điều chỉnh learning rate schedule
6. Áp dụng data augmentation cho các loại thực thể có recall thấp

## Kết luận

Bài thực hành đã xây dựng thành công mô hình RNN cho bài toán Named Entity Recognition trên bộ dữ liệu CoNLL-2003. Mô hình đạt 90.41% token-level accuracy và 59.85% entity-level F1-score, thể hiện khả năng tốt trong việc nhận dạng thực thể tên.

Kết quả cho thấy RNN đơn giản có thể phân loại token tốt nhưng còn hạn chế trong việc nhận dạng đầy đủ các thực thể phức tạp. Đặc biệt, pattern của PER có precision thấp cho thấy cần cải thiện khả năng phân biệt giữa các loại thực thể. Kiến trúc này phù hợp để hiểu cơ bản về NER, nhưng cần nâng cấp lên LSTM/Bi-LSTM hoặc Transformer để đạt hiệu suất cao hơn trong ứng dụng thực tế.
