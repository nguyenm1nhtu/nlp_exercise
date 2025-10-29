import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr, size
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisApp1").getOrCreate()

## --- Configurable parameters ---
MIN_FREQ = 5
MAX_FREQ = 900
MIN_SENT_LEN = 3
NUM_FEATURES = 3000

# 2. Load Data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
df = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Initial samples: {df.count()}")

# 3. Noise Filtering: Remove URLs, HTML tags, special characters, lowercase
clean_text = regexp_replace(lower(col("text")), r"https?://\S+|www\.\S+", "")
clean_text = regexp_replace(clean_text, r"<.*?>", "")
clean_text = regexp_replace(clean_text, r"[^a-zA-Z\s]", "")
df = df.withColumn("clean_text", clean_text)

# 4. Prepare label (convert label to int, tolerate malformed input)
df = df.withColumn("label", expr("try_cast(sentiment as int)"))
df = df.dropna(subset=["label", "clean_text"])
print(f"After dropna: {df.count()}")
df = df.filter(col("label").isin(-1, 1))
df = df.withColumn("label", expr("CASE WHEN label = -1 THEN 0 ELSE 1 END"))
print(f"After label filtering: {df.count()}")

## 5. Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = tokenizer.transform(df)
df = stopwordsRemover.transform(df)
print(f"After tokenization/stopword removal: {df.count()}")

## 6. Vocabulary Reduction: Remove rare/common words
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F
word_df = df.select(F.explode(col("filtered_words")).alias("word"))
word_freq = word_df.groupBy("word").count()
freq_words = word_freq.filter((col("count") >= MIN_FREQ) & (col("count") <= MAX_FREQ)).select("word")
freq_words_list = [row["word"] for row in freq_words.collect()]

def filter_tokens(tokens):
	return [t for t in tokens if t in freq_words_list]
from pyspark.sql.functions import udf
filter_tokens_udf = udf(filter_tokens, ArrayType(StringType()))
df = df.withColumn("filtered_words_final", filter_tokens_udf(col("filtered_words")))
print(f"After vocabulary reduction: {df.count()}")

# # Loại bỏ câu quá ngắn
# df = df.filter(size(col("filtered_words_final")) >= MIN_SENT_LEN)
# print(f"After removing short sentences: {df.count()}")

## 7. TF-IDF with reduced dimensionality
hashingTF = HashingTF(inputCol="filtered_words_final", outputCol="raw_features", numFeatures=NUM_FEATURES)
idf = IDF(inputCol="raw_features", outputCol="features")

# 8. Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 9. Build pipeline
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[hashingTF, idf, lr])

# 10. Train and evaluate
train_start = time.time()
model = pipeline.fit(train_df)
train_time = time.time() - train_start

eval_start = time.time()
predictions = model.transform(test_df)
eval_time = time.time() - eval_start

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)

# 11. Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_app1_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write(f"Model training time: {train_time:.4f} seconds\n")
	f.write(f"Model evaluation time: {eval_time:.4f} seconds\n")
	f.write(f"Test Accuracy: {accuracy:.4f}\n")
	f.write(f"Test F1 Score: {f1:.4f}\n")

# Also print to console
print(f"Model training time: {train_time:.4f} seconds")
print(f"Model evaluation time: {eval_time:.4f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

spark.stop()
