package com.mtu.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Normalizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import java.io.{File, PrintWriter}
// import com.mtu.spark.Utils._

object Lab17_NLPPipeline {
  
  // Helper function to calculate cosine similarity between two vectors
  def cosineSimilarity(v1: Vector, v2: Vector): Double = {
    val dotProduct = v1.toArray.zip(v2.toArray).map { case (a, b) => a * b }.sum
    val norm1 = math.sqrt(v1.toArray.map(x => x * x).sum)
    val norm2 = math.sqrt(v2.toArray.map(x => x * x).sum)
    if (norm1 == 0 || norm2 == 0) 0.0 else dotProduct / (norm1 * norm2)
  }
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // REQUEST 1: Customize document limit
    val limitDocuments = 1000 // Change this value to process different number of documents
    println(s"\n==> Processing $limitDocuments documents")

    // REQUEST 2: Detailed performance measurement - Read Data
    println("\n--- STAGE 1: Reading Dataset ---")
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val readStartTime = System.nanoTime()
    val initialDF = spark.read.json(dataPath).limit(limitDocuments)
    val recordCount = initialDF.count()
    val readDuration = (System.nanoTime() - readStartTime) / 1e9d
    println(f"✓ Successfully read $recordCount records in $readDuration%.2f seconds.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false)

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. --- Vectorization (Term Frequency) ---
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(50000)

    // 5. --- Vectorization (Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("tfidf_features")

    // REQUEST 3: Vector Normalization
    val normalizer = new Normalizer()
      .setInputCol(idf.getOutputCol)
      .setOutputCol("features")
      .setP(2.0) // L2 normalization

    // 6. --- Assemble the Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // REQUEST 2: Detailed performance measurement - Pipeline Fitting
    println("\n--- STAGE 2: Fitting the NLP Pipeline ---")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"✓ Pipeline fitting took $fitDuration%.2f seconds.")

    // REQUEST 2: Detailed performance measurement - Data Transformation
    println("\n--- STAGE 3: Transforming Data ---")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"✓ Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size
    val vocabStartTime = System.nanoTime()
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    val vocabDuration = (System.nanoTime() - vocabStartTime) / 1e9d
    println(f"✓ Vocabulary size calculation took $vocabDuration%.2f seconds.")
    println(s"✓ Actual vocabulary size: $actualVocabSize unique terms.")

    // --- Show Results ---
    println("\n--- STAGE 4: Sample of Transformed Data ---")
    transformedDF.select("text", "features").show(5, truncate = 50)

    // REQUEST 4: Find Similar Documents
    println("\n--- STAGE 5: Finding Similar Documents ---")
    val similarityStartTime = System.nanoTime()
    
    // Collect all documents with their features
    val allDocs = transformedDF
      .select("text", "features")
      .withColumn("doc_id", monotonically_increasing_id())
      .collect()
    
    // Select the first document as reference
    val referenceDoc = allDocs(0)
    val referenceId = referenceDoc.getAs[Long]("doc_id")
    val referenceText = referenceDoc.getAs[String]("text")
    val referenceVector = referenceDoc.getAs[Vector]("features")
    
    println(s"\n==> Reference Document (ID: $referenceId):")
    println(s"${referenceText.substring(0, Math.min(referenceText.length, 200))}...")
    
    // Calculate similarities
    val similarities = allDocs.map { row =>
      val docId = row.getAs[Long]("doc_id")
      val text = row.getAs[String]("text")
      val vector = row.getAs[Vector]("features")
      val similarity = if (docId == referenceId) 0.0 else cosineSimilarity(referenceVector, vector)
      (docId, text, similarity)
    }.filter(_._1 != referenceId) // Exclude the reference document itself
      .sortBy(-_._3) // Sort by similarity (descending)
      .take(5) // Top 5
    
    val similarityDuration = (System.nanoTime() - similarityStartTime) / 1e9d
    println(f"\n✓ Similarity calculation took $similarityDuration%.2f seconds.")
    
    println("\n==> Top 5 Most Similar Documents:")
    similarities.zipWithIndex.foreach { case ((docId, text, similarity), index) =>
      println(s"\n${index + 1}. Document ID: $docId (Similarity: ${similarity}%.4f)")
      println(s"   Text: ${text.substring(0, Math.min(text.length, 150))}...")
    }

    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)

    // REQUEST 2: Detailed performance measurement - Write Metrics
    println("\n--- STAGE 6: Writing Output Files ---")
    val writeStartTime = System.nanoTime()

    // Write metrics to the log folder
    val log_path = "../log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("=" * 80)
      logWriter.println("NLP PIPELINE PERFORMANCE METRICS")
      logWriter.println("=" * 80)
      logWriter.println(s"Document Limit: $limitDocuments")
      logWriter.println(s"Actual Documents Processed: $recordCount")
      logWriter.println()
      logWriter.println("--- Stage Execution Times ---")
      logWriter.println(f"1. Data Reading:           $readDuration%.2f seconds")
      logWriter.println(f"2. Pipeline Fitting:       $fitDuration%.2f seconds")
      logWriter.println(f"3. Data Transformation:    $transformDuration%.2f seconds")
      logWriter.println(f"4. Vocabulary Calculation: $vocabDuration%.2f seconds")
      logWriter.println(f"5. Similarity Calculation: $similarityDuration%.2f seconds")
      logWriter.println(f"Total Processing Time:     ${readDuration + fitDuration + transformDuration + vocabDuration + similarityDuration}%.2f seconds")
      logWriter.println()
      logWriter.println("--- Pipeline Configuration ---")
      logWriter.println(s"Tokenizer: RegexTokenizer")
      logWriter.println(s"Stop Words Removal: Enabled")
      logWriter.println(s"HashingTF numFeatures: 50000")
      logWriter.println(s"IDF: Enabled")
      logWriter.println(s"Normalizer: L2 Normalization (Enabled)")
      logWriter.println()
      logWriter.println("--- Vocabulary Statistics ---")
      logWriter.println(s"Actual vocabulary size: $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures: 50000")
      if (50000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (50000) < vocabulary size ($actualVocabSize)")
        logWriter.println("      Hash collisions are expected.")
      }
      logWriter.println()
      logWriter.println("--- Similar Documents Analysis ---")
      logWriter.println(s"Reference Document ID: $referenceId")
      logWriter.println(s"Reference Text: ${referenceText.substring(0, Math.min(referenceText.length, 100))}...")
      logWriter.println("\nTop 5 Most Similar Documents:")
      similarities.zipWithIndex.foreach { case ((docId, text, similarity), index) =>
        logWriter.println(f"${index + 1}. Document $docId (Similarity: $similarity%.4f)")
        logWriter.println(s"   ${text.substring(0, Math.min(text.length, 100))}...")
      }
      logWriter.println()
      logWriter.println("=" * 80)
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("For detailed stage-level metrics, view Spark UI at http://localhost:4040")
      logWriter.println("=" * 80)
      println(s"✓ Successfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Write data results to the results folder
    val result_path = "../results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println("=" * 80)
      resultWriter.println(s"NLP PIPELINE OUTPUT (First $n_results results)")
      resultWriter.println("=" * 80)
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val features = row.getAs[Vector]("features")
        resultWriter.println("=" * 80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Normalized TF-IDF Vector (first 10 dimensions): ${features.toArray.take(10).mkString(", ")}...")
        resultWriter.println(s"Vector size: ${features.size}")
        resultWriter.println("=" * 80)
        resultWriter.println()
      }
      println(s"✓ Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }

    val writeDuration = (System.nanoTime() - writeStartTime) / 1e9d
    println(f"✓ File writing took $writeDuration%.2f seconds.")

    // Final Summary
    val totalDuration = readDuration + fitDuration + transformDuration + vocabDuration + similarityDuration + writeDuration
    println("\n" + "=" * 80)
    println("EXECUTION SUMMARY")
    println("=" * 80)
    println(f"Total Execution Time: $totalDuration%.2f seconds")
    println(s"Documents Processed: $recordCount")
    println(s"Vocabulary Size: $actualVocabSize")
    println("Pipeline Stages: Tokenization → Stop Words Removal → TF → IDF → Normalization")
    println("=" * 80)

    spark.stop()
    println("\nSpark Session stopped.")
  }
}