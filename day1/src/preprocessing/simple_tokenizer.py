from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()

        for punc in [".", ",", "?", "!"]:
            text = text.replace(punc, f" {punc} ")

        tokens = text.split()

        return tokens