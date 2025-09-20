from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

def test_tokenizers():
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    for s in sentences:
        print("SimpleTokenizer:", simple_tokenizer.tokenize(s))
        print("RegexTokenizer :", regex_tokenizer.tokenize(s), "\n")

if __name__ == "__main__":
    test_tokenizers()
