# Lab 1 & Lab 2

## 1. Mô tả công việc

### Lab 1

- Cài đặt interface `Tokenizer` (abstract base class) trong `src/core/interfaces.py`.
- Cài đặt `SimpleTokenizer` trong `src/preprocessing/simple_tokenizer.py`:
  - Chuyển text về lowercase, tách token theo whitespace và tách các dấu câu cơ bản (.,?!).
- Cài đặt `RegexTokenizer` trong `src/preprocessing/regex_tokenizer.py`:
  - Sử dụng regex (`\w+|[^\w\s]`) để tách từ và dấu câu, xử lý tốt hơn các trường hợp đặc biệt.

### Lab 2

- Cài đặt interface `Vectorizer` trong `src/core/interfaces.py` với các method: `fit`, `transform`, `fit_transform`.
- Cài đặt `CountVectorizer` trong `src/representations/count_vectorizer.py`:
  - Nhận một tokenizer, xây dựng vocabulary từ corpus, chuyển văn bản thành vector đếm.
- Viết các file test trong `src/test/` để kiểm thử các tokenizer và vectorizer.
- Cài đặt hàm `load_raw_text_data` trong `src/core/dataset_loaders.py` để load dữ liệu từ file UD English EWT.

## 2. Cách chạy code và ghi log kết quả

- Để kiểm thử các chức năng của Lab 1 và Lab 2, chạy các file test sau:
  ```
  python test/main.py      # kiểm thử tokenizer (Lab 1)
  python test/lab2_test.py # kiểm thử CountVectorizer (Lab 2)
  ```
- Kết quả sẽ in ra tokenization, vocabulary và document-term matrix cho cả corpus mẫu và 5 dòng đầu của dataset UD English EWT.

## 3. Kết quả chạy code

### Output của các tokenizer trên các câu mẫu:

```
SimpleTokenizer Results:
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

RegexTokenizer Results:
Input: Hello, world! This is a test.
Tokens: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
Tokens: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
Tokens: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

### Output của các tokenizer trên 100 kí tự của UD English EWT:

```
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...

SimpleTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### Output của CountVectorizer trên corpus mẫu:

```
Learned vocabulary:
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-term matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

### Output của CountVectorizer trên 5 dòng đầu của UD English EWT:

```
[UD English EWT] Learned vocabulary (first 5 lines):
{'!': 0, ',': 1, '-': 2, '.': 3, '2': 4, '3': 5, ':': 6, '[': 7, ']': 8, 'a': 9, 'abdullah': 10, 'al': 11, 'american': 12, 'ani': 13, 'announced': 14, 'at': 15, 'authorities': 16, 'baghdad': 17, 'be': 18, 'being': 19, 'border': 20, 'busted': 21, 'by': 22, 'causing': 23, 'cells': 24, 'cleric': 25, 'come': 26, 'dpa': 27, 'for': 28, 'forces': 29, 'had': 30, 'in': 31, 'interior': 32, 'iraqi': 33, 'killed': 34, 'killing': 35, 'ministry': 36, 'moi': 37, 'mosque': 38, 'near': 39, 'of': 40, 'officials': 41, 'operating': 42, 'preacher': 43, 'qaim': 44, 'respected': 45, 'run': 46, 'shaikh': 47, 'syrian': 48, 'terrorist': 49, 'that': 50, 'the': 51, 'them': 52, 'they': 53, 'this': 54, 'to': 55, 'town': 56, 'trouble': 57, 'two': 58, 'up': 59, 'us': 60, 'were': 61, 'will': 62, 'years': 63, 'zaman': 64}

[UD English EWT] Document-term matrix (first 5 lines):
[0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```

## 4. Giải thích kết quả

- **SimpleTokenizer** tách các dấu câu cơ bản, nhưng có thể tách chưa tốt với các ký tự đặc biệt hoặc contraction.
- **RegexTokenizer** dùng regex nên tách tốt hơn, đặc biệt với các dấu câu liền nhau hoặc các ký tự không phải chữ/số.
- **CountVectorizer** xây dựng vocabulary từ toàn bộ corpus, mỗi document được biểu diễn thành vector đếm số lần xuất hiện của từng từ trong vocabulary.
- Khi áp dụng lên dataset lớn như UD English EWT, số chiều của vector tăng mạnh, vocabulary đa dạng hơn.

## 5. Khó khăn và cách giải quyết

- **Vấn đề import module/package:** Khi chạy file test cần chú ý cấu trúc package và đường dẫn, có thể phải thêm sys.path hoặc chạy bằng `python -m ...` từ thư mục gốc.
- **Xử lý tiếng Anh tự nhiên:** Một số trường hợp contraction, ký tự đặc biệt, hoặc dấu câu liên tiếp vẫn có thể gây lỗi tách token, cần tinh chỉnh regex hoặc logic tokenizer.

---
