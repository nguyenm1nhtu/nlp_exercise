import sys
import os
import re
import time
from datetime import datetime
from gensim.models import Word2Vec
import numpy as np
from typing import List

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src', 'representations')
sys.path.insert(0, src_dir)


class EWTDataStreamer:
    """Data loader for EWT corpus."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_sentences(self, output_file=None) -> List[List[str]]:
        """Load sentences from EWT corpus file."""
        if not os.path.exists(self.file_path):
            error_msg = f"Error: File not found: {self.file_path}"
            if output_file:
                output_file.write(error_msg + "\n")
            return []
            
        sentences = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = re.findall(r'\b[a-zA-Z]+\b', line.lower())
                    if len(tokens) >= 3:
                        sentences.append(tokens)
        
        return sentences


def train_word2vec_model(data_path: str, model_save_path: str, output_file) -> Word2Vec:
    """Train a Word2Vec model on the EWT corpus."""
    
    # Load training data
    output_file.write("Loading training data...\n")
    streamer = EWTDataStreamer(data_path)
    sentences = streamer.load_sentences(output_file)
    
    if not sentences:
        output_file.write("No training data available!\n")
        return None
    
    output_file.write(f"Training on {len(sentences)} sentences...\n")
    
    # Record training start time
    training_start_time = time.time()
    output_file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10
    )
    
    # Record training end time
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    output_file.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)\n")
    output_file.write(f"Training completed. Vocabulary: {len(model.wv.key_to_index):,} words\n")
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    output_file.write(f"Model saved to: {model_save_path}\n")
    
    return model


def demonstrate_model_usage(model: Word2Vec, output_file):
    """Demonstrate the usage of the trained Word2Vec model."""
    
    output_file.write(f"\nModel demo - Vocab: {len(model.wv.key_to_index):,}, Dims: {model.wv.vector_size}\n")
    
    # Test words
    test_words = ["the", "man", "woman", "king", "good", "bad"]
    available_words = [word for word in test_words if word in model.wv.key_to_index]
    
    if len(available_words) >= 2:
        # Word similarities
        output_file.write("Similarities:\n")
        for i in range(min(2, len(available_words)-1)):
            word1, word2 = available_words[i], available_words[i+1]
            try:
                similarity = model.wv.similarity(word1, word2)
                output_file.write(f"  {word1}-{word2}: {similarity:.3f}\n")
            except KeyError:
                pass
    
    # Most similar words
    if available_words:
        word = available_words[0]
        try:
            similar = model.wv.most_similar(word, topn=3)
            output_file.write(f"Similar to '{word}': {', '.join([w for w, s in similar])}\n")
        except KeyError:
            pass
    
    # Analogy
    if "man" in model.wv.key_to_index and "woman" in model.wv.key_to_index and "king" in model.wv.key_to_index:
        try:
            result = model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
            if result:
                answer, score = result[0]
                output_file.write(f"Analogy man:woman :: king:{answer} ({score:.3f})\n")
        except:
            pass


def main():
    """Main function to run the embedding training demo."""
    
    # Setup output file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file_path = os.path.join(results_dir, 'lab4_training_demo_output.txt')
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("Lab 4: Custom Word2Vec Training Demo\n")
        output_file.write(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 60 + "\n\n")
        
        # File paths
        data_path = os.path.join("data", "UD_English-EWT", "en_ewt-ud-train.txt")
        model_save_path = os.path.join("results", "word2vec_ewt.model")
        
        if not os.path.exists(data_path):
            output_file.write(f"Error: Training data not found at {data_path}\n")
            return
        
        try:
            # Record total execution start time
            total_start_time = time.time()
            
            # Train the model
            model = train_word2vec_model(data_path, model_save_path, output_file)
            
            if model:
                # Demonstrate usage
                demonstrate_model_usage(model, output_file)
                
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
    
    print(f"Output saved to: {output_file_path}")


if __name__ == "__main__":
    main()