import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisApp2").getOrCreate()

# 2. Load Data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.withColumn("clean_text", col("text"))

## 3. Prepare label 
df = df.withColumn("label", expr("try_cast(sentiment as int)"))
df = df.dropna(subset=["label", "clean_text"])
df = df.filter(col("label").isin(-1, 1))
df = df.withColumn("label", expr("CASE WHEN label = -1 THEN 0 ELSE 1 END"))

## 4. Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = tokenizer.transform(df)
df = stopwordsRemover.transform(df)

## 5. Word2Vec Embedding 
word2vec = Word2Vec(vectorSize=50, minCount=3, inputCol="filtered_words", outputCol="w2v_features")

# 6. Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 7. Build pipeline
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="w2v_features", labelCol="label")
pipeline = Pipeline(stages=[word2vec, lr])

# 8. Train and evaluate
train_start = time.time()
model = pipeline.fit(train_df)
train_time = time.time() - train_start

eval_start = time.time()
predictions = model.transform(test_df)
eval_time = time.time() - eval_start

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)

# 10. Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_app2_results.txt')
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
