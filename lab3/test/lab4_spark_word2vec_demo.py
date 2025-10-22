import os
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, size
from pyspark.ml.feature import Word2Vec, StopWordsRemover, Tokenizer
import pyspark.sql.functions as F

def create_spark_session(output_file):
    """Initialize Spark session."""
    spark = SparkSession.builder \
        .appName("Lab4_PySpark_Word2Vec") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    output_file.write("Spark initialized\n")
    return spark

def load_and_preprocess_data(spark, data_path, output_file, sample_fraction=1.0):
    """Load and preprocess C4 JSON dataset."""
    output_file.write("Loading data...\n")
    
    # Read JSON file and sample data
    df = spark.read.json(data_path).sample(fraction=sample_fraction, seed=42)
    
    # Clean and tokenize
    cleaned_df = df.select(
        regexp_replace(
            regexp_replace(lower(col("text")), r"[^\w\s]", ""), 
            r"\s+", " "
        ).alias("cleaned_text")
    ).filter(F.length(col("cleaned_text")) > 50)
    
    # Tokenize and remove stop words
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="raw_words")
    stop_remover = StopWordsRemover(inputCol="raw_words", outputCol="words")
    
    processed_df = stop_remover.transform(tokenizer.transform(cleaned_df))
    final_df = processed_df.filter(size(col("words")) >= 3).select("words")
    
    count = final_df.count()
    output_file.write(f"Processed {count:,} documents\n")
    return final_df

def train_word2vec_model(df, output_file, vector_size=100, min_count=3, max_iter=5):
    """Train Word2Vec model."""
    output_file.write("Training Word2Vec model...\n")
    
    # Record training start time
    training_start_time = time.time()
    output_file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    word2vec = Word2Vec(
        inputCol="words", 
        outputCol="features",
        vectorSize=vector_size,
        minCount=min_count,
        maxIter=max_iter,
        seed=42
    )
    
    model = word2vec.fit(df)
    
    # Record training end time
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    output_file.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)\n")
    
    return model

def demonstrate_model_usage(model, output_file, test_word="computer", top_n=5):
    """Demonstrate model capabilities."""
    vectors = model.getVectors()
    vocab_size = vectors.count()
    
    output_file.write(f"Vocabulary: {vocab_size:,} words\n")
    
    try:
        similar_words = model.findSynonymsArray(test_word, top_n)
        output_file.write(f"Similar to '{test_word}':\n")
        for i, (word, sim) in enumerate(similar_words, 1):
            output_file.write(f"  {i}. {word} ({sim:.3f})\n")
    except:
        output_file.write(f"'{test_word}' not in vocabulary\n")

def save_model_results(model, output_file):
    """Save vocabulary results."""
    try:
        os.makedirs("results", exist_ok=True)
        vectors = model.getVectors()
        vocab_list = [row["word"] for row in vectors.select("word").collect()]
        
        with open("results/spark_vocab_simple.txt", 'w', encoding='utf-8') as f:
            for word in sorted(vocab_list):
                f.write(f"{word}\n")
        
        output_file.write(f"Vocabulary saved: {len(vocab_list):,} words\n")
    except Exception as e:
        output_file.write(f"Save error: {e}\n")

def main():
    """Main function for PySpark Word2Vec training."""
    
    # Setup output file
    os.makedirs("results", exist_ok=True)
    output_file_path = os.path.join("results", 'lab4_spark_word2vec_output.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("Lab 4: PySpark Word2Vec Training\n")
        output_file.write(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 60 + "\n\n")
        
        data_path = "data/c4-train.00000-of-01024-30K.json"
        
        spark = None
        try:
            # Record total execution start time
            total_start_time = time.time()
            
            # Initialize and train
            spark = create_spark_session(output_file)
            
            if not os.path.exists(data_path):
                output_file.write(f"Data file not found: {data_path}\n")
                return
            
            # Process data and train model (using full dataset: sample_fraction=1.0)
            processed_df = load_and_preprocess_data(spark, data_path, output_file, sample_fraction=1.0)
            processed_df.cache()
            
            model = train_word2vec_model(processed_df, output_file, vector_size=100)
            
            # Test similarity
            demonstrate_model_usage(model, output_file, test_word="computer", top_n=5)
            
            # Try alternative words
            test_words = ["data", "system", "technology", "the"]
            for word in test_words:
                try:
                    similar = model.findSynonymsArray(word, 3)
                    if similar:
                        output_file.write(f"Similar to '{word}': {', '.join([w for w, s in similar[:3]])}\n")
                        break
                except:
                    continue
            
            # Save results
            save_model_results(model, output_file)
            
            # Record total execution time
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            
            output_file.write("\n" + "=" * 60 + "\n")
            output_file.write("SUMMARY\n")
            output_file.write("=" * 60 + "\n")
            output_file.write(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
            output_file.write(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output_file.write("Training completed successfully!\n")
            
        except Exception as e:
            output_file.write(f"Error: {e}\n")
            output_file.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        finally:
            if spark:
                spark.stop()
    
    print(f"Output saved to: {output_file_path}")

if __name__ == "__main__":
    main()