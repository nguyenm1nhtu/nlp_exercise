# Lab 2: Building a Spark NLP Pipeline

## 1. Các bước triển khai

Môi trường: Scala 2.12, Spark 3.5.1 (MLlib), SBT 1.11.6, OpenJDK 17.

Dữ liệu: C4 Common Crawl (30K records, xử lý 1,000 records để demo).

Pipeline:

- Đọc dữ liệu JSON → Spark DataFrame
- Tách từ: RegexTokenizer (pattern: \\s+|[.,;!?()\"'])
- Loại bỏ stop words: StopWordsRemover
- Tần suất từ: HashingTF (50,000 features)
- TF-IDF: IDF
- Chuẩn hóa: Normalizer
- Tính độ tương đồng: Cosine similarity

## 2. Cách chạy và log kết quả

### Chạy code:

```python
cd E:\NLP\spark_labs
sbt compile
sbt "runMain com.lhson.spark.Lab17_NLPPipeline"
```

### Theo dõi kết quả:

Console: hiển thị progress.

Spark UI: http://localhost:4040.

Log file: log/lab17_metrics.log.

Output file: results/lab17_pipeline_output.txt.

### Tùy chỉnh cấu hình:

```python
val limitDocuments = 1000  // Thay đổi số lượng documents
val numFeatures = 50000    // Thay đổi kích thước vector
```

## 3. Kết quả đạt được

Hiệu năng từng giai đoạn:

| Giai đoạn         | Thời gian  | Tỷ lệ    |
| ----------------- | ---------- | -------- |
| Đọc dữ liệu       | 5.14s      | 28.9%    |
| Fit Pipeline      | 2.90s      | 16.3%    |
| Transform dữ liệu | 1.32s      | 7.4%     |
| Tính Vocabulary   | 0.62s      | 3.5%     |
| Tính Similarity   | 7.82s      | 43.9%    |
| **Tổng**          | **17.79s** | **100%** |

Thống kê:

- 1,000 records được xử lý.

- Kích thước vocabulary: 31,355 từ duy nhất

- Feature vector: 50,000 chiều

- Hash collisions: 0% (50K features > 31K vocab)

Ý nghĩa: TF-IDF vectors phản ánh tầm quan trọng của từ và có thể sử dụng cho các bài toán ML tiếp theo.

## 4. Khó khăn và cách giải quyết

1. Tương thích phiên bản Java

Vấn đề: Spark 3.5.1 yêu cầu Java 17+
Giải pháp: Sử dụng OpenJDK 17 LTS, cấu hình JAVA_HOME đúng

2. Quản lý bộ nhớ

Vấn đề: OutOfMemory với dataset lớn
Giải pháp:

Tăng driver memory: sbt -J-Xmx4g run
Cache DataFrame sau transform
Giới hạn số documents xử lý

3. Hash Collisions

Vấn đề: 20K features < 31K vocab → mất thông tin
Giải pháp: Tăng lên 50K features (không còn collision, chất lượng tốt hơn)
Trade-off: Thời gian tính similarity tăng +3.6s

4. Môi trường Windows

Vấn đề: Thiếu winutils.exe, cảnh báo liên tục
Giải pháp: Bỏ qua cảnh báo (không ảnh hưởng) hoặc tải winutils

## 5. Công cụ và mô hình sử dụng

MLlib components: RegexTokenizer, StopWordsRemover, HashingTF, IDF.
