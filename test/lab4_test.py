import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src', 'representations')
sys.path.insert(0, src_dir)

from word_embedder import WordEmbedder

def main():
    """Main test function."""
    
    # Setup output file
    results_dir = os.path.join(os.path.dirname(current_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, 'lab4_test_output.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Lab 4: WordEmbedder Test\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Initialize WordEmbedder
        try:
            embedder = WordEmbedder('glove-wiki-gigaword-50')
        except Exception as e:
            f.write(f"Failed to load model: {e}\n")
            return
    
        # 1. Get vector for 'king'
        f.write("1. Get vector for 'king':\n")
        king_vector = embedder.get_vector('king')
        if king_vector is not None:
            f.write(f"Shape: {king_vector.shape}\n")
            f.write(f"First 5 values: {king_vector[:5]}\n")
        else:
            f.write("'king' not found\n")
        
        # 2. Word similarities
        f.write("\n2. Word similarities:\n")
        king_queen_sim = embedder.get_similarity('king', 'queen')
        king_man_sim = embedder.get_similarity('king', 'man')
        
        if king_queen_sim and king_man_sim:
            f.write(f"king-queen: {king_queen_sim:.4f}\n")
            f.write(f"king-man: {king_man_sim:.4f}\n")
        
        # 3. Most similar words to 'computer'
        f.write("\n3. Most similar to 'computer':\n")
        similar_words = embedder.get_most_similar('computer', top_n=10)
        for i, (word, sim) in enumerate(similar_words[:5], 1):
            f.write(f"{i}. {word} ({sim:.3f})\n")
        
        # 4. Document embedding
        f.write("\n4. Document embedding:\n")
        sentence = "The queen rules the country."
        doc_vector = embedder.embed_document(sentence)
        f.write(f"Sentence: '{sentence}'\n")
        f.write(f"Vector shape: {doc_vector.shape}\n")
        f.write(f"First 5 values: {doc_vector[:5]}\n")
        f.write(f"Is zero vector: {np.allclose(doc_vector, 0)}\n")
        
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        # Also save error to file
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        error_file = os.path.join(results_dir, 'lab4_test_error.txt')
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {e}\n")
        print(f"Error log saved to: {error_file}")