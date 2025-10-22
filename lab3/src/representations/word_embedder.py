import gensim.downloader as api
from typing import List, Tuple, Optional
import numpy as np


class WordEmbedder:
    """
    A class for loading and working with pre-trained word embedding models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the WordEmbedder with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model (e.g., 'glove-wiki-gigaword-50')
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}...")
        self.model = api.load(model_name)
        print(f"Model loaded successfully! Vocabulary size: {len(self.model.key_to_index):,}")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Returns the embedding vector for a given word.
        Handles Out-of-Vocabulary (OOV) words.
        
        Args:
            word (str): The word to get vector for
            
        Returns:
            np.ndarray: Vector representation of the word, or None if word not found (OOV)
        """
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f"Warning: '{word}' is not in vocabulary (OOV word)")
            return None
    
    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Returns the cosine similarity between the vectors of two words.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: Cosine similarity between the two words, or None if either word is OOV
        """
        if word1 not in self.model.key_to_index:
            print(f"Warning: '{word1}' is not in vocabulary (OOV word)")
            return None
        if word2 not in self.model.key_to_index:
            print(f"Warning: '{word2}' is not in vocabulary (OOV word)")
            return None
        
        return self.model.similarity(word1, word2)
    
    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Uses the model's built-in most_similar method to find the top N most similar words.
        
        Args:
            word (str): The target word
            top_n (int): Number of most similar words to return (default: 10)
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity_score) tuples, 
                                    or empty list if word is OOV
        """
        if word not in self.model.key_to_index:
            print(f"Warning: '{word}' is not in vocabulary (OOV word)")
            return []
        
        return self.model.most_similar(word, topn=top_n)
    
    def word_exists(self, word: str) -> bool:
        """
        Check if a word exists in the vocabulary.
        
        Args:
            word (str): The word to check
            
        Returns:
            bool: True if word exists in vocabulary, False otherwise
        """
        return word in self.model.key_to_index
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the model's vocabulary.
        
        Returns:
            int: Number of words in vocabulary
        """
        return len(self.model.key_to_index)
    
    def get_vector_dimension(self) -> int:
        """
        Get the dimension of the word vectors.
        
        Returns:
            int: Dimension of word vectors
        """
        return self.model.vector_size

    def embed_document(self, document: str) -> np.ndarray:
        """
        Embed a full document by averaging known word vectors.

        Steps:
        - Tokenize the input document using a Tokenizer from Lab1 if available.
        - For each token, retrieve its vector via `get_vector` and ignore OOV words.
        - If no in-vocabulary tokens are found, return a zero vector of the model's dimension.
        - Otherwise compute the element-wise mean of all token vectors and return it.

        Args:
            document (str): The input document text to embed.

        Returns:
            np.ndarray: 1D array with the document embedding of shape (vector_dim,)
        """
        # Try to import the Lab1 tokenizer (best-effort). If not available, fall back to a simple regex split.
        tokens: List[str]
        try:
            # Lab1 tokenizer module path used in the workspace (Lab1/src/preprocessing/simple_tokenizer.py)
            from Lab1.src.preprocessing.simple_tokenizer import RegexTokenizer, SimpleTokenizer
            tokenizer = RegexTokenizer()
            tokens = tokenizer.tokenize(document)
        except Exception:
            # Fallback: simple whitespace / punctuation-based tokenizer
            import re
            tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", document.lower())]

        vecs: List[np.ndarray] = []
        for tok in tokens:
            v = self.get_vector(tok)
            if v is not None:
                vecs.append(np.asarray(v, dtype=float))

        dim = self.get_vector_dimension()
        if len(vecs) == 0:
            # Return zero vector if no known words
            return np.zeros(dim, dtype=float)

        # Compute element-wise mean
        stacked = np.vstack(vecs)
        doc_vec = np.mean(stacked, axis=0)
        return doc_vec