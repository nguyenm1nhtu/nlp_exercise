import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
from sklearn.model_selection import train_test_split
from Lab1.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab1.src.representations.count_vectorizer import CountVectorizer
from Lab4.src.models.text_classifier import TextClassifier

texts = [
	"This movie is fantastic and I love it!",
	"I hate this film, it's terrible.",
	"The acting was superb, a truly great experience.",
	"What a waste of time, absolutely boring.",
	"Highly recommend this, a masterpiece.",
	"Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Instantiate tokenizer and vectorizer
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer)

# Instantiate classifier
clf = TextClassifier(vectorizer)


# Measure training and prediction time
start_time = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - start_time

start_pred_time = time.time()
y_pred = clf.predict(X_test)
pred_time = time.time() - start_pred_time

# Evaluate
metrics = clf.evaluate(y_test, y_pred)

# Prepare output
output_lines = []
output_lines.append(f"Model training time: {train_time:.4f} seconds")
output_lines.append(f"Model prediction time: {pred_time:.4f} seconds")
output_lines.append("Evaluation metrics:")
for k, v in metrics.items():
	output_lines.append(f"{k}: {v:.3f}")

# Save to results folder
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'lab5_test_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write('\n'.join(output_lines))

# Also print to console
for line in output_lines:
	print(line)
