# Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

## I. Mục tiêu Lab

Lab này tập trung vào việc xây dựng và so sánh 4 pipeline phân loại văn bản khác nhau trên dataset HWU-64 (Intent Classification):

1. **TF-IDF + Logistic Regression** (Baseline 1)
2. **Word2Vec (Averaging) + Dense Layer** (Baseline 2)
3. **Embedding (Pre-trained) + LSTM** 
4. **Embedding (Học từ đầu) + LSTM**

Mục đích chính là hiểu rõ ưu điểm của các mô hình chuỗi (RNN/LSTM) trong việc nắm bắt ngữ cảnh và thứ tự từ, so với các phương pháp truyền thống như Bag-of-Words.

---

## II. Kết quả So sánh Định lượng

### Bảng so sánh các mô hình trên tập Test

| Pipeline | Accuracy | F1-score (Macro) | Test Loss |
|---------|----------|------------------|-----------|
| **TF-IDF + Logistic Regression** | **83.36%** | **83.00%** | N/A |
| **Word2Vec (Avg) + Dense** | 58.83% | 54.78% | 1.4335 |
| **Embedding (Pre-trained) + LSTM** | 53.07% | 50.44% | 1.6899 |
| **Embedding (Scratch) + LSTM** | 3.53% | 0.23% | 3.8935 |

### Nhận xét về kết quả định lượng

Kết quả cho thấy mô hình đơn giản nhất (TF-IDF + Logistic Regression) đạt hiệu suất tốt nhất với **83.36% accuracy**, trong khi các mô hình LSTM phức tạp hơn lại cho kết quả kém hơn nhiều.

**Phân tích nguyên nhân:**

1. **TF-IDF + Logistic Regression (83.36% - Tốt nhất):**
   - Đơn giản, ổn định, ít bị overfitting
   - Phù hợp với dataset có vocabulary không quá lớn
   - Logistic Regression có khả năng tổng quát hóa tốt với dữ liệu văn bản ngắn
   - **N-grams** (1,2) giúp bắt được một số pattern cục bộ quan trọng

2. **Word2Vec (Avg) + Dense (58.83% - Trung bình):**
   - Mất thông tin về **thứ tự từ** do averaging
   - Word2Vec được train trên dataset nhỏ (chỉ train set) nên chất lượng embedding chưa cao
   - Mô hình Dense đơn giản chưa đủ capacity để học tốt từ averaged vectors

3. **Embedding (Pre-trained) + LSTM (53.07% - Kém):**
   - Pre-trained embeddings bị **đóng băng** (trainable=False) - không thể adapt với domain cụ thể
   - Word2Vec được train trên training set nhỏ, không phải corpus lớn như GloVe/FastText
   - LSTM có thể bị **underfitting** do không đủ epochs hoặc learning rate chưa tối ưu
   - Dataset có thể quá nhỏ để LSTM phát huy ưu thế

4. **Embedding (Scratch) + LSTM (3.53% - Thất bại hoàn toàn):**
   - **Catastrophic failure** - accuracy chỉ 3.53% (worse than random guessing ~1.56% for 64 classes)
   - Mô hình bị **underfitting nghiêm trọng** hoặc không học được gì
   - Có thể do:
     - Learning rate không phù hợp (quá cao hoặc quá thấp)
     - Batch size quá lớn (256) khiến gradient không ổn định
     - Embedding layer chưa được khởi tạo tốt
     - Dataset quá nhỏ để học cả embedding và LSTM từ đầu

**Kết luận:**

> **Mô hình phức tạp không phải lúc nào cũng tốt hơn!** Với dataset nhỏ và task đơn giản, các mô hình đơn giản như TF-IDF + Logistic Regression có thể hoạt động tốt hơn các deep learning models phức tạp. LSTM cần dataset lớn hơn, hyperparameter tuning cẩn thận, và pre-trained embeddings chất lượng cao để phát huy ưu thế.

---

## III. Phân tích Định tính - Dự đoán trên Các Câu Khó

### Các câu test với phủ định và cấu trúc phức tạp

#### **Câu 1: "can you remind me to not call my mom"**
- **Phân loại đúng:** `calendar_reminder` hoặc `alarm_set` (câu có ý nghĩa tạo lời nhắc)
- **Kết quả:**
  - TF-IDF + LR: `calendar_set` (gần đúng)
  - Word2Vec + Dense: `social_post` (sai)
  - LSTM (Pre-trained): `social_post` (sai)
  - LSTM (Scratch): `recommendation_events` (sai)

**Phân tích:** 
- Câu này chứa **phủ định "not"** đứng trước động từ "call", tạo thành cụm "to not call"
- TF-IDF bắt được pattern **"remind me to"** - một bigram quan trọng cho intent reminder
- Các mô hình khác (đặc biệt LSTM) không hiểu được ngữ cảnh phủ định, nhầm lẫn với social_post vì chứa từ "mom"
- LSTM lý thuyết nên tốt hơn nhưng thực tế lại kém vì chưa được train đủ tốt

---

#### **Câu 2: "is it going to be sunny or rainy tomorrow"**
- **Phân loại đúng:** `weather_query`
- **Kết quả:**
  - TF-IDF + LR: `weather_query` (đúng)
  - Word2Vec + Dense: `weather_query` (đúng)
  - LSTM (Pre-trained): `weather_query` (đúng)
  - LSTM (Scratch): `recommendation_events` (sai)

**Phân tích:**
- Câu này không có phủ định, chỉ có liên từ "or" tạo sự lựa chọn
- **3/4 mô hình dự đoán đúng** vì từ khóa "sunny", "rainy", "tomorrow" rất đặc trưng cho weather_query
- TF-IDF bắt được bigrams như "sunny or", "or rainy", "rainy tomorrow"
- Ngay cả Word2Vec averaging cũng đủ vì các từ weather-related có **semantic similarity** cao
- LSTM Scratch vẫn thất bại do không học được gì

---

#### **Câu 3: "find a flight from new york to london but not through paris"**
- **Phân loại đúng:** `transport_query` hoặc `transport_flight`
- **Kết quả:**
  - TF-IDF + LR: `transport_query` (đúng)
  - Word2Vec + Dense: `recommendation_locations` (sai)
  - LSTM (Pre-trained): `social_post` (sai)
  - LSTM (Scratch): `recommendation_events` (sai)

**Phân tích:**
- Câu này phức tạp nhất với:
  - **Phụ thuộc xa**: "find a flight" ... "but not through paris"
  - Cấu trúc phủ định: "but not through"
  - Nhiều địa danh: new york, london, paris
- Chỉ TF-IDF dự đoán đúng nhờ bắt được **n-grams**:
  - "find a flight" (trigram cực mạnh cho transport)
  - "from new", "to london" (patterns của transport queries)
- Word2Vec + Dense nhầm lẫn vì averaging vectors của các địa danh → semantic gần "recommendation_locations"
- LSTM lý thuyết nên tốt hơn vì có thể nhớ "find a flight" từ đầu câu, nhưng thực tế:
  - Pre-trained LSTM chưa học được **dependency dài**
  - Embeddings không đủ tốt để encode ý nghĩa transport

---

#### **Câu 4: "don't forget to turn off the lights"**
- **Phân loại đúng:** `iot_hue_lightoff`
- **Kết quả:**
  - TF-IDF + LR: `iot_hue_lightoff` (đúng)
  - Word2Vec + Dense: `iot_hue_lightoff` (đúng)
  - LSTM (Pre-trained): `iot_hue_lightoff` (đúng)
  - LSTM (Scratch): `iot_hue_lightoff` (đúng)

**Phân tích:**
- **Tất cả 4 mô hình đều đúng**
- Câu này có phủ định "don't" nhưng không gây nhầm lẫn vì:
  - Pattern **"turn off the lights"** cực kỳ đặc trưng và rõ ràng
  - Không có từ nhiễu hoặc ngữ cảnh phức tạp
  - Ngay cả LSTM Scratch (đang thất bại ở mọi câu khác) cũng đoán đúng
- Đây là câu "dễ" mà mọi mô hình đều handle được

---

#### **Câu 5: "I want to order pizza but not with pepperoni"**
- **Phân loại đúng:** `takeaway_order`
- **Kết quả:**
  - TF-IDF + LR: `takeaway_order` (đúng)
  - Word2Vec + Dense: `takeaway_order` (đúng)
  - LSTM (Pre-trained): `iot_coffee` (sai)
  - LSTM (Scratch): `recommendation_events` (sai)

**Phân tích:**
- Câu có phủ định **"but not with"** để chỉ điều không muốn
- TF-IDF và Word2Vec đúng vì:
  - Bigram **"order pizza"** cực mạnh cho takeaway_order
  - "I want to order" là pattern rõ ràng
  - Phủ định ở cuối câu không ảnh hưởng nhiều đến intent chính
- LSTM Pre-trained sai lầm nghiêm trọng: dự đoán `iot_coffee`
  - Có thể mô hình nhầm lẫn giữa "order pizza" và "make coffee" (cả 2 đều là food/beverage)
  - Embedding chưa tốt nên không phân biệt được takeaway vs iot
- LSTM Scratch: Vẫn dự đoán `recommendation_events` như mọi khi (**mode collapse**)

---

### Tổng kết phân tích định tính

#### Các câu LSTM lý thuyết nên tốt hơn (nhưng thực tế không phải):

1. **Câu 1** - "remind me to **not** call my mom" → phủ định trong cụm động từ
2. **Câu 3** - "find a flight ... **but not through** paris" → phụ thuộc xa + phủ định
3. **Câu 5** - "order pizza **but not with** pepperoni" → phủ định chi tiết

#### Tại sao LSTM không hoạt động tốt như kỳ vọng?

**Lý thuyết:**
- LSTM được thiết kế để xử lý **long-term dependencies** qua gates mechanism
- Có thể nhớ thông tin từ đầu câu (ví dụ: "find a flight") khi xử lý cuối câu
- Hidden state được cập nhật tuần tự qua từng token, nắm bắt ngữ cảnh

**Thực tế trong lab này:**
1. **Dataset quá nhỏ:** HWU-64 có ~9k training samples, chưa đủ cho LSTM học tốt
2. **Pre-trained embeddings chất lượng thấp:**
   - Word2Vec chỉ train trên training set nhỏ
   - Không phải GloVe/FastText trained trên billions of tokens
   - Bị đóng băng (trainable=False) nên không adapt được
3. **Hyperparameters chưa tối ưu:**
   - Batch size 256 quá lớn cho dataset nhỏ
   - Learning rate mặc định có thể không phù hợp
   - Dropout 0.2 có thể chưa đủ để tránh overfitting
4. **Training không đủ epochs:**
   - EarlyStopping có thể dừng quá sớm
   - LSTM cần nhiều epochs hơn để converge
5. **LSTM Scratch thất bại hoàn toàn:**
   - Học đồng thời embedding + sequential patterns quá khó
   - Cần dataset lớn hơn rất nhiều

**Kết luận:**
> LSTM có tiềm năng cao hơn cho các câu phức tạp, **NHƯNG** cần điều kiện đủ tốt: dataset lớn, pre-trained embeddings chất lượng cao (GloVe/FastText/BERT), hyperparameter tuning cẩn thận, và training đủ lâu. Với dataset nhỏ như HWU-64, các mô hình đơn giản như TF-IDF + Logistic Regression có thể là lựa chọn tốt hơn.

---

## IV. Nhận xét chung về Ưu và Nhược điểm của từng phương pháp

### 1. TF-IDF + Logistic Regression

**Ưu điểm:**
- Đơn giản, nhanh, dễ triển khai - không cần GPU
- Hiệu suất tốt với dataset nhỏ/trung bình (như trong lab này: 83.36%)
- Ổn định, ít bị overfitting - không cần tuning nhiều
- **Interpretable** - có thể xem feature importance
- **N-grams** capture local patterns - bigrams/trigrams bắt được một số context
- Không cần pre-training - chỉ cần raw text

**Nhược điểm:**
- Không hiểu **thứ tự từ** - "not good" và "good not" giống nhau
- Không hiểu **semantic** - "excellent" và "good" là 2 features độc lập
- **Sparse vectors** - chiều cao (thousands of features), lãng phí bộ nhớ
- Không generalize ra từ mới (out-of-vocabulary)
- Yếu với câu dài, phức tạp - không nhớ được dependencies xa

**Khi nào nên dùng:**
- Dataset nhỏ/trung bình (<100k samples)
- Cần triển khai nhanh, không có GPU
- Task đơn giản, văn bản ngắn (tweets, queries, headlines)
- Cần model interpretable cho business

---

### 2. Word2Vec (Averaging) + Dense Layer

**Ưu điểm:**
- **Dense vectors** - chiều thấp (100-300d), tiết kiệm bộ nhớ
- Capture **semantic similarity** - "good" và "excellent" có vectors gần nhau
- Pre-trained embeddings có thể dùng (GloVe, FastText)
- Handle OOV tốt hơn (nếu dùng FastText với subword)
- Nhanh hơn LSTM - chỉ cần averaging + feedforward

**Nhược điểm:**
- Mất hoàn toàn thông tin **thứ tự** - averaging làm mất cấu trúc câu
- Từ "not" bị **neutralize** khi average với các từ khác
- Không nhớ được context - mỗi từ độc lập
- Yếu với câu dài - càng nhiều từ, vector trung bình càng generic
- Cần Word2Vec chất lượng - train trên corpus nhỏ sẽ kém

**Khi nào nên dùng:**
- Cần embedding-based approach nhưng không có GPU/thời gian cho LSTM
- Task phụ thuộc vào từ khóa hơn là cấu trúc (ví dụ: topic classification)
- Có sẵn pre-trained embeddings tốt (GloVe, FastText)
- Dataset trung bình, không quá phức tạp

---

### 3. Embedding (Pre-trained) + LSTM

**Ưu điểm (Lý thuyết):**
- Xử lý **sequences** - hiểu thứ tự từ, context
- **Long-term dependencies** - nhớ thông tin xa qua hidden state
- Pre-trained embeddings - khởi đầu tốt hơn
- Phù hợp với câu dài, phức tạp - có cơ chế gates để kiểm soát thông tin
- Tốt với **phủ định** - có thể học pattern "not good" khác "good"

**Nhược điểm (Thực tế trong lab):**
- Cần dataset lớn - HWU-64 quá nhỏ (9k samples)
- Pre-trained embeddings quality matters - Word2Vec train trên train set chưa tốt
- Đóng băng embeddings (trainable=False) → không adapt được domain
- Chậm, cần GPU - training time lâu hơn nhiều
- Nhiều hyperparameters - khó tune (learning rate, batch size, dropout, etc.)
- Dễ overfit hoặc underfit - cần validation set và callbacks
- Kết quả trong lab: chỉ 53% - kém hơn TF-IDF

**Khi nào nên dùng:**
- Dataset lớn (>50k samples)
- Có GPU và thời gian training
- Có pre-trained embeddings chất lượng cao (GloVe, FastText, BERT)
- Task phức tạp: sentiment with negation, question answering, sequence labeling
- Câu dài, cấu trúc phức tạp, phụ thuộc xa

**Cách cải thiện trong tương lai:**
1. Dùng **GloVe hoặc FastText** thay vì Word2Vec train trên dataset nhỏ
2. **Unfreeze embeddings** sau vài epochs (fine-tuning)
3. Tăng dataset qua data augmentation
4. Tune hyperparameters: learning rate, batch size, LSTM units
5. Thử **Bidirectional LSTM** để xem context 2 chiều

---

### 4. Embedding (Scratch) + LSTM

**Ưu điểm (Lý thuyết):**
- Embeddings học đặc thù cho task - không bị constraint bởi pre-trained
- Linh hoạt nhất - mọi thứ đều trainable
- Không cần pre-trained - end-to-end learning
- Tốt nhất khi có dataset khổng lồ (millions of samples)

**Nhược điểm (Thực tế trong lab - THẤT BẠI):**
- **Accuracy 3.53%** - thất bại hoàn toàn
- Cần dataset cực lớn - phải học cả embeddings lẫn LSTM từ đầu
- Training rất khó - dễ bị **vanishing/exploding gradient**
- Chậm converge - cần rất nhiều epochs
- Dễ bị **mode collapse** - mô hình dự đoán cùng 1 class cho mọi input
- Cần tuning rất cẩn thận:
  - Learning rate phải vừa đủ
  - Batch size không được quá lớn/nhỏ
  - Initialization quan trọng
  - **Gradient clipping** để tránh exploding

**Tại sao thất bại trong lab:**
1. Dataset quá nhỏ (9k samples) → không đủ học 128*100 + LSTM parameters
2. Batch size 256 quá lớn → gradient không stable
3. Không có gradient clipping → có thể bị exploding gradient
4. Learning rate mặc định có thể không phù hợp
5. **Cold start problem** - embeddings khởi tạo random, mất nhiều epochs mới có nghĩa

**Khi nào nên dùng:**
- Dataset khổng lồ (>500k samples, ideally millions)
- Task rất specific, không có pre-trained phù hợp
- Có GPU mạnh và thời gian training lâu
- Có kinh nghiệm tuning hyperparameters

**Không nên dùng:**
- Dataset nhỏ như HWU-64
- Không có GPU
- Cần kết quả nhanh

---

## V. Kết luận và Bài học

### Bài học quan trọng nhất từ lab này:

> **"Simpler is often better"** - Mô hình phức tạp (LSTM) không phải lúc nào cũng tốt hơn mô hình đơn giản (TF-IDF + Logistic Regression). Success depends on matching model complexity to data size and task complexity.

### Thứ tự ưu tiên khi chọn mô hình:

1. **Small dataset (<10k):** TF-IDF + Logistic Regression
2. **Medium dataset (10k-50k):** 
   - Thử TF-IDF trước (baseline)
   - Nếu cần semantic: Word2Vec/GloVe + Dense
   - Nếu task phức tạp: LSTM với pre-trained GloVe/FastText chất lượng cao
3. **Large dataset (>50k):**
   - LSTM với pre-trained embeddings
   - Fine-tune embeddings
4. **Huge dataset (>500k):**
   - LSTM train from scratch có thể cạnh tranh
   - Hoặc dùng Transformer (BERT, RoBERTa)

### Để LSTM hoạt động tốt hơn trong tương lai:

1. **Tăng dataset**: Data augmentation, thu thập thêm data
2. **Pre-trained embeddings tốt**: GloVe 300d, FastText, hoặc contextual (BERT)
3. **Hyperparameter tuning**:
   - Learning rate: thử 0.001, 0.0005, 0.0001
   - Batch size: thử 32, 64 (nhỏ hơn 256)
   - LSTM units: thử 64, 128, 256
   - Dropout: thử 0.3, 0.4, 0.5
4. **Advanced techniques**:
   - Bidirectional LSTM
   - Attention mechanism
   - Gradient clipping (clipnorm=1.0)
   - Learning rate scheduling
5. **Regularization**: L2, dropout, early stopping với patience lớn hơn

### Khi nào chuyển sang Transformer (BERT)?

Nếu:
- Dataset > 10k và có label chất lượng
- Cần state-of-the-art performance
- Có GPU mạnh (>=8GB VRAM)
- Task phức tạp (sentiment + aspect, NER, QA)

→ Dùng **fine-tuned BERT** thay vì LSTM

---

## VI. Tài liệu tham khảo

---

## VII. Source Code

Toàn bộ source code được lưu trong notebook: `lab5_rnn_text_classification.ipynb`

**Cấu trúc:**
- Cells 1-10: Data loading và preprocessing
- Cells 11-12: Task 1 - TF-IDF + Logistic Regression
- Cells 13-19: Task 2 - Word2Vec + Dense
- Cells 20-27: Task 3 - Embedding (Pre-trained) + LSTM
- Cells 28-32: Task 4 - Embedding (Scratch) + LSTM
- Cells 33-40: Task 5 - Evaluation và Analysis

---

