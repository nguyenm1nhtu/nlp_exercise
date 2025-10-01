package com.lhson.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
import org.apache.log4j.{Logger, Level}
import java.text.SimpleDateFormat
import java.util.Date
// import com.lhson.spark.Utils._

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .config("spark.ui.port", "4040")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") // Use Java serializer for Word2Vec compatibility
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    // Set log level to INFO for our application
    spark.sparkContext.setLogLevel("ERROR")

    // Custom logging configuration
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.INFO)

    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // Read the C4 dataset into a Spark DataFrame
    val df = spark.read.json("data/c4-train.00000-of-01024-30K.json.gz")
      .limit(1000) // Limit to 1000 records for faster processing

    println(s"Successfully read ${df.count()} records.")
    df.printSchema()
    println("Sample of initial DataFrame:")
    df.show(5, truncate = false)

    // --- Implement a Spark ML Pipeline ---

    // Use RegexTokenizer or Tokenizer for tokenization
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W") // Split on non-word characters

    // Use StopWordsRemover to remove stop words
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    // Use HashingTF and IDF for vectorization
    val hashingTF = new HashingTF()
      .setInputCol("filtered_words")
      .setOutputCol("tf_features")
      .setNumFeatures(20000) // Number of features in the hash table

    val idf = new IDF()
      .setInputCol("tf_features")
      .setOutputCol("features")

    // Create and fit the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))

    // Fit the pipeline and transform the data
    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(df)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(df)
    transformedDF.cache() // Cache the result for efficiency
    val transformCount = transformedDF.count() // Force an action to trigger the transformation
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_words").as("word"))
      .filter(length($"word") > 1) // Filter out single-character tokens
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // --- Show and Save Results ---
    println("\nSample of transformed data:")
    transformedDF.select("text", "words", "filtered_words", "features").show(5, truncate = 50)

    println("\nDataFrame Schema After Transformation:")
    transformedDF.printSchema()

    println("\nFeature vector information:")
    val featuresInfo = transformedDF.select("features").first()
    val featuresVector = featuresInfo.getAs[org.apache.spark.ml.linalg.Vector]("features")
    println(s"--> Feature vector size: ${featuresVector.size}")
    println(s"--> Feature vector type: ${featuresVector.getClass.getSimpleName}")

    val n_results = 20
    val results = transformedDF.select("text", "words", "filtered_words", "features").take(n_results)

    // Log the process - Write metrics to the log folder
    val log_path = "log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs() // Ensure directory exists
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- NLP Pipeline Processing Log ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF feature vector size: ${featuresVector.size}")
      logWriter.println(s"Records processed: $transformCount")
      logWriter.println(s"Log file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      println(s"\nSuccessfully wrote log to $log_path")
    } finally {
      logWriter.close()
    }

    // Save the results to a file
    val result_path = "results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs() // Ensure directory exists
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val words = row.getAs[Seq[String]]("words")
        val filteredWords = row.getAs[Seq[String]]("filtered_words")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Tokenized Words: ${words.take(10).mkString(", ")}...")
        resultWriter.println(s"Filtered Words: ${filteredWords.take(10).mkString(", ")}...")
        resultWriter.println(s"Feature Vector Size: ${features.size}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }

    spark.stop()
    println("Spark Session stopped.")
  }
}
