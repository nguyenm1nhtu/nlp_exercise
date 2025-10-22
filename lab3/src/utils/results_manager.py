"""
Results Manager for organizing and analyzing experiment results.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsManager:
    """Manager for analyzing and visualizing saved results."""
    
    def __init__(self, results_path: str = "results"):
        """
        Initialize ResultsManager.
        
        Args:
            results_path (str): Path to results directory
        """
        self.results_path = results_path
        self.results_dir = os.path.join(results_path, "results")
        self.reports_dir = os.path.join(results_path, "reports")
        
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def load_results(self, filename: str) -> Dict:
        """
        Load results from JSON file.
        
        Args:
            filename (str): Name of the results file
            
        Returns:
            Dict: Loaded results
        """
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    
    def get_similarity_results(self) -> pd.DataFrame:
        """
        Get all similarity results as DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with similarity results
        """
        similarity_files = [f for f in os.listdir(self.results_dir) 
                           if f.startswith('similarity_') and f.endswith('.json')]
        
        all_results = []
        for filename in similarity_files:
            results = self.load_results(filename)
            for word, score in results['similar_words']:
                all_results.append({
                    'target_word': results['target_word'],
                    'model': results['model'],
                    'similar_word': word,
                    'similarity_score': score,
                    'timestamp': results['timestamp']
                })
        
        return pd.DataFrame(all_results)
    
    def compare_models_similarity(self, target_word: str) -> pd.DataFrame:
        """
        Compare similarity results across different models for a target word.
        
        Args:
            target_word (str): The target word to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        df = self.get_similarity_results()
        word_results = df[df['target_word'] == target_word]
        
        if word_results.empty:
            print(f"No results found for word: {target_word}")
            return pd.DataFrame()
        
        # Pivot table for comparison
        comparison = word_results.pivot_table(
            index='similar_word', 
            columns='model', 
            values='similarity_score', 
            fill_value=0
        )
        
        return comparison
    
    def plot_similarity_comparison(self, target_word: str, top_n: int = 10):
        """
        Plot similarity comparison across models.
        
        Args:
            target_word (str): Target word
            top_n (int): Number of top similar words to show
        """
        comparison = self.compare_models_similarity(target_word)
        
        if comparison.empty:
            return
        
        # Get top N words by average similarity
        comparison['avg_similarity'] = comparison.mean(axis=1)
        top_words = comparison.nlargest(top_n, 'avg_similarity')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Remove average column for plotting
        plot_data = top_words.drop('avg_similarity', axis=1)
        
        # Create bar plot
        ax = plot_data.plot(kind='bar', width=0.8)
        plt.title(f'Top {top_n} Similar Words to "{target_word}" Across Models')
        plt.xlabel('Similar Words')
        plt.ylabel('Similarity Score')
        plt.legend(title='Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, f'similarity_comparison_{target_word}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved: {plot_path}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all experiments.
        
        Returns:
            str: Path to the generated report
        """
        # Get all results
        df = self.get_similarity_results()
        
        if df.empty:
            print("No results found to generate report")
            return ""
        
        # Generate summary statistics
        summary = {
            "total_experiments": len(df['target_word'].unique()),
            "models_tested": df['model'].unique().tolist(),
            "target_words": df['target_word'].unique().tolist(),
            "average_similarity_by_model": df.groupby('model')['similarity_score'].mean().to_dict(),
            "experiments_by_model": df['model'].value_counts().to_dict(),
            "generation_time": datetime.now().isoformat()
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"summary_report_{timestamp}.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        md_report_path = os.path.join(self.reports_dir, f"summary_report_{timestamp}.md")
        self._generate_markdown_report(summary, df, md_report_path)
        
        print(f"Summary report saved: {report_path}")
        print(f"Markdown report saved: {md_report_path}")
        
        return md_report_path
    
    def _generate_markdown_report(self, summary: Dict, df: pd.DataFrame, filepath: str):
        """Generate markdown report."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Word Embedding Experiment Summary Report\n\n")
            f.write(f"**Generated:** {summary['generation_time']}\n\n")
            
            f.write("## Overview\n")
            f.write(f"- **Total Experiments:** {summary['total_experiments']}\n")
            f.write(f"- **Models Tested:** {', '.join(summary['models_tested'])}\n")
            f.write(f"- **Target Words:** {', '.join(summary['target_words'])}\n\n")
            
            f.write("## Model Performance\n")
            f.write("### Average Similarity Scores by Model\n")
            for model, avg_score in summary['average_similarity_by_model'].items():
                f.write(f"- **{model}:** {avg_score:.4f}\n")
            
            f.write("\n### Experiment Count by Model\n")
            for model, count in summary['experiments_by_model'].items():
                f.write(f"- **{model}:** {count} experiments\n")
            
            f.write("\n## Detailed Results\n")
            f.write("### Top Similar Words by Target\n")
            
            for target_word in summary['target_words']:
                f.write(f"\n#### Target Word: '{target_word}'\n")
                word_data = df[df['target_word'] == target_word]
                
                for model in summary['models_tested']:
                    model_data = word_data[word_data['model'] == model]
                    if not model_data.empty:
                        f.write(f"\n**{model} Model:**\n")
                        top_5 = model_data.nlargest(5, 'similarity_score')
                        for _, row in top_5.iterrows():
                            f.write(f"- {row['similar_word']}: {row['similarity_score']:.4f}\n")
    
    def export_results_csv(self) -> str:
        """
        Export all results to CSV file.
        
        Returns:
            str: Path to exported CSV file
        """
        df = self.get_similarity_results()
        
        if df.empty:
            print("No results to export")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.reports_dir, f"all_results_{timestamp}.csv")
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Results exported to CSV: {csv_path}")
        
        return csv_path