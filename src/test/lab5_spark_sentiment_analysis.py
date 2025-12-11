
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Load Data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
df = spark.read.csv(data_path, header=True, inferSchema=True)
# Normalize sentiment labels (-1/1 to 0/1)
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
# Drop rows with null sentiment values
df = df.dropna(subset=["sentiment"])

# Split data into train/test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 3. Build Preprocessing Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 4. Train the Model with timing
train_start = time.time()
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
model = pipeline.fit(train_df)
train_time = time.time() - train_start

# 5. Evaluate the Model with timing
eval_start = time.time()
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)
eval_time = time.time() - eval_start

# Prepare output
output_lines = []
output_lines.append(f"Model training time: {train_time:.4f} seconds")
output_lines.append(f"Model evaluation time: {eval_time:.4f} seconds")
output_lines.append(f"Test Accuracy: {accuracy:.4f}")
output_lines.append(f"Test F1 Score: {f1:.4f}")

# Save to results folder
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write('\n'.join(output_lines))

# Also print to console
for line in output_lines:
	print(line)