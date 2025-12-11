import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, expr, size
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F

# --- Configurable parameters ---
MIN_FREQ = 5
MAX_FREQ = 1000
MIN_SENT_LEN = 3
NUM_FEATURES = 2000

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysisAdvanced").getOrCreate()

# 2. Load Data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiments.csv'))
df = spark.read.csv(data_path, header=True, inferSchema=True)

# 3. Noise Filtering
clean_text = regexp_replace(lower(col("text")), r"https?://\S+|www\.\S+", "")
clean_text = regexp_replace(clean_text, r"<.*?>", "")
clean_text = regexp_replace(clean_text, r"[^a-zA-Z\s]", "")
df = df.withColumn("clean_text", clean_text)

# 4. Prepare label
df = df.withColumn("label", expr("try_cast(sentiment as int)"))
df = df.dropna(subset=["label", "clean_text"])
df = df.filter(col("label").isin(-1, 1))
df = df.withColumn("label", expr("CASE WHEN label = -1 THEN 0 ELSE 1 END"))

# 5. Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = tokenizer.transform(df)
df = stopwordsRemover.transform(df)

# 6. Vocabulary Reduction
word_df = df.select(F.explode(col("filtered_words")).alias("word"))
word_freq = word_df.groupBy("word").count()
freq_words = word_freq.filter((col("count") >= MIN_FREQ) & (col("count") <= MAX_FREQ)).select("word")
freq_words_list = [row["word"] for row in freq_words.collect()]
def filter_tokens(tokens):
	return [t for t in tokens if t in freq_words_list]
filter_tokens_udf = F.udf(filter_tokens, ArrayType(StringType()))
df = df.withColumn("filtered_words_final", filter_tokens_udf(col("filtered_words")))
df = df.filter(size(col("filtered_words_final")) >= MIN_SENT_LEN)

# 7. Feature Extraction
# TF-IDF
hashingTF = HashingTF(inputCol="filtered_words_final", outputCol="raw_features", numFeatures=NUM_FEATURES)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
# Word2Vec
word2vec = Word2Vec(vectorSize=100, minCount=2, inputCol="filtered_words_final", outputCol="w2v_features")

# 8. Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

results = []
def run_pipeline(stages, model, train_df, test_df, featuresCol):
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

# 9. Models to run
models = [
	("LogisticRegression", LogisticRegression(maxIter=10, regParam=0.001, featuresCol="tfidf_features", labelCol="label"), [hashingTF, idf], "tfidf_features"),
	("NaiveBayes", NaiveBayes(featuresCol="tfidf_features", labelCol="label"), [hashingTF, idf], "tfidf_features"),
	("GBT", GBTClassifier(featuresCol="tfidf_features", labelCol="label", maxIter=10), [hashingTF, idf], "tfidf_features"),
	("NeuralNet", MultilayerPerceptronClassifier(featuresCol="tfidf_features", labelCol="label", maxIter=20, layers=[NUM_FEATURES, 64, 64, 2], blockSize=128, seed=42), [hashingTF, idf], "tfidf_features"),
	("LogisticRegression_W2V", LogisticRegression(maxIter=10, regParam=0.001, featuresCol="w2v_features", labelCol="label"), [word2vec], "w2v_features"),
	("GBT_W2V", GBTClassifier(featuresCol="w2v_features", labelCol="label", maxIter=10), [word2vec], "w2v_features"),
	("NeuralNet_W2V", MultilayerPerceptronClassifier(featuresCol="w2v_features", labelCol="label", maxIter=20, layers=[100, 64, 64, 2], blockSize=128, seed=42), [word2vec], "w2v_features"),
]

# 10. Run all models
for name, model, stages, featuresCol in models:
	train_time, eval_time, accuracy, f1 = run_pipeline(stages, model, train_df, test_df, featuresCol)
	results.append((name, train_time, eval_time, accuracy, f1))

# 11. Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_spark_sentiment_advanced_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write("Model\tTrainTime(s)\tEvalTime(s)\tAccuracy\tF1\n")
	for name, train_time, eval_time, accuracy, f1 in results:
		f.write(f"{name}\t{train_time:.2f}\t{eval_time:.2f}\t{accuracy:.4f}\t{f1:.4f}\n")

# Print results
for name, train_time, eval_time, accuracy, f1 in results:
	print(f"{name}: Train {train_time:.2f}s, Eval {eval_time:.2f}s, Acc {accuracy:.4f}, F1 {f1:.4f}")

spark.stop()
