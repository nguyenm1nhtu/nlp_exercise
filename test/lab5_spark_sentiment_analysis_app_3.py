import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import NaiveBayes, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisApp3").getOrCreate()

# 2. Load Data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
df = spark.read.csv(data_path, header=True, inferSchema=True)

df = df.withColumn("clean_text", col("text"))

# 3. Prepare label (convert sentiment to int, tolerate malformed input)
df = df.withColumn("label", expr("try_cast(sentiment as int)"))
df = df.dropna(subset=["label", "clean_text"])
df = df.filter(col("label").isin(-1, 1))
df = df.withColumn("label", expr("CASE WHEN label = -1 THEN 0 ELSE 1 END"))

## 4. Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = tokenizer.transform(df)
df = stopwordsRemover.transform(df)

## 5. Bag-of-Words features
hashingTF = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=2000)

# 6. Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


results = []

def run_pipeline(stages, model, train_df, test_df):
    pipeline = Pipeline(stages=stages + [model])
    start_train = time.time()
    model_fit = pipeline.fit(train_df)
    train_time = time.time() - start_train
    start_eval = time.time()
    predictions = model_fit.transform(test_df)
    eval_time = time.time() - start_eval
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(predictions)
    return train_time, eval_time, accuracy, f1


# 7. Naive Bayes
nb = NaiveBayes(featuresCol="features", labelCol="label")
train_time, eval_time, accuracy, f1 = run_pipeline([hashingTF], nb, train_df, test_df)
results.append(("Bag-of-Words + NaiveBayes", train_time, eval_time, accuracy, f1))

# 8. Gradient-Boosted Trees
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)
train_time, eval_time, accuracy, f1 = run_pipeline([hashingTF], gbt, train_df, test_df)
results.append(("Bag-of-Words + GBT", train_time, eval_time, accuracy, f1))

# 9. Neural Network (MultilayerPerceptronClassifier)
# 2 lớp ẩn, mỗi lớp 64 node, đầu ra 2 lớp (binary)
layers = [2000, 64, 64, 2]
mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", maxIter=20, layers=layers, blockSize=128, seed=42)
train_time, eval_time, accuracy, f1 = run_pipeline([hashingTF], mlp, train_df, test_df)
results.append(("Bag-of-Words + NeuralNet", train_time, eval_time, accuracy, f1))

# 10. Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_app3_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("Model\tTrainTime(s)\tEvalTime(s)\tAccuracy\tF1\n")
    for name, train_time, eval_time, accuracy, f1 in results:
        f.write(f"{name}\t{train_time:.2f}\t{eval_time:.2f}\t{accuracy:.4f}\t{f1:.4f}\n")

# Print results
for name, train_time, eval_time, accuracy, f1 in results:
    print(f"{name}: Train {train_time:.2f}s, Eval {eval_time:.2f}s, Acc {accuracy:.4f}, F1 {f1:.4f}")

spark.stop()
