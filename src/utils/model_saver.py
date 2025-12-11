"""
Model Saver utility for saving and loading word embedding models and results.
"""

import pickle
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from datetime import datetime


class ModelSaver:
    """Utility class for saving and loading models and results."""
    
    def __init__(self, base_path: str = "results"):
        """
        Initialize ModelSaver with base directory path.
        
        Args:
            base_path (str): Base directory to save models and results
        """
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, "models")
        self.results_dir = os.path.join(base_path, "results")
        self.logs_dir = os.path.join(base_path, "logs")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: The model to save
            model_name (str): Name for the saved model
            metadata (Dict): Additional metadata about the model
            
        Returns:
            str: Path where the model was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            metadata_file = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: The loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded: {filepath}")
        return model
    
    def save_results(self, results: Dict, experiment_name: str) -> str:
        """
        Save experiment results.
        
        Args:
            results (Dict): Results dictionary to save
            experiment_name (str): Name of the experiment
            
        Returns:
            str: Path where results were saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved: {filepath}")
        return filepath
    
    def save_similarity_results(self, word: str, similar_words: List[Tuple[str, float]], 
                              model_name: str) -> str:
        """
        Save word similarity results.
        
        Args:
            word (str): Target word
            similar_words (List[Tuple[str, float]]): List of similar words with scores
            model_name (str): Name of the model used
            
        Returns:
            str: Path where results were saved
        """
        results = {
            "target_word": word,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "similar_words": similar_words
        }
        
        filename = f"similarity_{word}_{model_name}"
        return self.save_results(results, filename)
    
    def save_vector_data(self, word: str, vector: np.ndarray, model_name: str) -> str:
        """
        Save word vector data.
        
        Args:
            word (str): The word
            vector (np.ndarray): Word vector
            model_name (str): Name of the model
            
        Returns:
            str: Path where vector was saved
        """
        results = {
            "word": word,
            "vector": vector.tolist(),
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "vector_size": len(vector)
        }
        
        filename = f"vector_{word}_{model_name}"
        return self.save_results(results, filename)
    
    def save_log(self, message: str, log_type: str = "info") -> None:
        """
        Save log message to file.
        
        Args:
            message (str): Log message
            log_type (str): Type of log (info, error, warning)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{log_type.upper()}] {message}\n"
        
        log_file = os.path.join(self.logs_dir, "experiment.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def list_saved_models(self) -> List[str]:
        """List all saved models."""
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        return sorted(model_files)
    
    def list_saved_results(self) -> List[str]:
        """List all saved results."""
        result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        return sorted(result_files)