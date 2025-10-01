
## 1. Các bước triển khai

Môi trường: Scala 2.12, Spark 3.5.1 (MLlib), SBT 1.11.6, OpenJDK 17.

Dữ liệu: C4 Common Crawl (30K records, xử lý 1,000 records để demo).

Pipeline:

- Đọc dữ liệu JSON vào DataFrame.

- Tiền xử lý: RegexTokenizer → StopWordsRemover.

- Vector hóa: HashingTF (20,000 features) → IDF.

- Fit pipeline và transform dữ liệu.

- Xuất kết quả ra file và ghi log.

## 2. Cách chạy và log kết quả

### Chạy code:

cd E:\NLP\spark_labs
sbt compile
sbt "runMain com.lhson.spark.Lab17_NLPPipeline"


### Theo dõi kết quả:

Console: hiển thị progress.

Spark UI: http://localhost:4040.

Log file: log/lab17_metrics.log.

Output file: results/lab17_pipeline_output.txt.

## 3. Kết quả đạt được

Thống kê:

- 1,000 records được xử lý.

- Thời gian fitting ~3.26s, transform ~1.18s.

- Vocabulary size sau preprocessing: 27,009.

- Feature vector: 20,000 chiều.

Ý nghĩa: TF-IDF vectors phản ánh tầm quan trọng của từ và có thể sử dụng cho các bài toán ML tiếp theo.

## 4. Khó khăn và cách giải quyết

Java version: một số lỗi tương thích → dùng OpenJDK 17 LTS và thay Word2Vec bằng HashingTF + IDF.

Memory: tăng driver memory (-J-Xmx4g) và điều chỉnh numFeatures để giảm hash collision.

Windows environment: bỏ qua cảnh báo winutils, dùng relative path.

Hiệu năng: cold start Spark lâu → cache DataFrame và bật adaptive query execution.

## 5. Công cụ và mô hình sử dụng

MLlib components: RegexTokenizer, StopWordsRemover, HashingTF, IDF.

