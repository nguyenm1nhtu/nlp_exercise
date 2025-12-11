# Lab 5: PyTorch Introduction
---

## Phần 1: Khám phá Tensor

### Task 1.1: Tạo và Khám phá Tensor

**Mục tiêu**: Làm quen với cách tạo tensor trong PyTorch từ nhiều nguồn khác nhau.

**Output**:
```
Tensor from data:
tensor([[1, 2],
        [3, 4]])

Tensor from NumPy array:
tensor([[1, 2],
        [3, 4]], dtype=torch.int32)

Ones Tensor:
tensor([[1, 1],
        [1, 1]])

Random Tensor:
tensor([[0.7358, 0.9234],
        [0.4567, 0.2890]])

Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu
```

**Phân tích kết quả**:
- **Tensor from data**: Tạo thành công từ list Python, dtype mặc định là `torch.int64` cho số nguyên
- **Tensor from NumPy**: Giữ nguyên dtype của NumPy array (`int32`), cho thấy sự tương thích giữa 2 thư viện
- **Ones Tensor**: Shape được kế thừa từ `x_data` là (2, 2), tất cả giá trị đều là 1
- **Random Tensor**: Các giá trị ngẫu nhiên trong khoảng [0, 1), dtype được chuyển thành `float32` như đã chỉ định
- **Device = cpu**: Mặc định tensor được lưu trên CPU, có thể chuyển sang GPU bằng `.to('cuda')` nếu có

**Nhận xét**: 
- PyTorch hỗ trợ nhiều cách tạo tensor linh hoạt
- Có thể chuyển đổi dễ dàng giữa NumPy và PyTorch
- Các thuộc tính tensor (shape, dtype, device) rất quan trọng khi làm việc với deep learning

---

### Task 1.2: Các Phép Toán Trên Tensor

**Mục tiêu**: Thực hiện các phép toán cơ bản và phép toán ma trận trên tensor.

**Output**:
```
Cộng x_data với chính nó:
tensor([[2, 4],
        [6, 8]])

Nhân x_data với 5:
tensor([[ 5, 10],
        [15, 20]])

Nhân ma trận x_data với x_data.T:
tensor([[ 5, 11],
        [11, 25]])
```

**Phân tích từng phép toán**:

1. **Phép cộng element-wise** (`x_data + x_data`):
   - Ma trận gốc: [[1, 2], [3, 4]]
   - Kết quả: [[1+1, 2+2], [3+3, 4+4]] = [[2, 4], [6, 8]]
   - Đây là phép cộng từng phần tử tương ứng

2. **Phép nhân với scalar** (`x_data * 5`):
   - Broadcasting: scalar 5 được nhân với mỗi phần tử
   - Kết quả: [[1×5, 2×5], [3×5, 4×5]] = [[5, 10], [15, 20]]
   - Thường dùng để scale dữ liệu

3. **Phép nhân ma trận** (`x_data @ x_data.T`):
   - x_data = [[1, 2], [3, 4]], x_data.T = [[1, 3], [2, 4]]
   - Vị trí [0,0]: (1×1 + 2×2) = 5
   - Vị trí [0,1]: (1×3 + 2×4) = 11
   - Vị trí [1,0]: (3×1 + 4×2) = 11
   - Vị trí [1,1]: (3×3 + 4×4) = 25
   - Kết quả là ma trận đối xứng vì nhân với chuyển vị của chính nó

**Nhận xét**:
- PyTorch hỗ trợ cả phép toán element-wise (`+`, `*`) và phép toán ma trận (`@`)
- Toán tử `@` tương đương với `torch.matmul()`
- Các phép toán được tối ưu hóa tốt cho hiệu suất cao

---

### Task 1.3: Indexing và Slicing

**Mục tiêu**: Truy xuất các phần tử, hàng, cột từ tensor.

**Output**:
```
Hàng đầu tiên của x_data: tensor([1, 2])
Cột thứ hai của x_data: tensor([2, 4])
Giá trị ở hàng 2, cột 2 của x_data: tensor(4)
```

**Phân tích chi tiết**:

1. **Lấy hàng đầu tiên** (`x_data[0]`):
   - Ma trận gốc: [[1, 2], [3, 4]]
   - Index 0 → hàng đầu tiên → [1, 2]
   - Kết quả là tensor 1D với 2 phần tử
   - Shape giảm từ (2, 2) xuống (2,)

2. **Lấy cột thứ hai** (`x_data[:, 1]`):
   - Ký hiệu `:` nghĩa là "tất cả các hàng"
   - Index 1 ở chiều thứ 2 → cột thứ hai
   - Lấy phần tử thứ 2 của mỗi hàng: [2 (từ hàng 1), 4 (từ hàng 2)]
   - Kết quả: tensor([2, 4])

3. **Lấy giá trị cụ thể** (`x_data[1, 1]`):
   - Index [1, 1] = hàng thứ 2, cột thứ 2
   - Ma trận: [[1, 2], [3, **4**]]
   - Kết quả: tensor(4) - vẫn là tensor chứ không phải số Python
   - Dùng `.item()` để chuyển thành số: `x_data[1, 1].item()` → 4

**Bảng minh họa indexing**:
```
Ma trận x_data:
        Cột 0   Cột 1
Hàng 0:   1       2
Hàng 1:   3       4

x_data[0]    → [1, 2]      (hàng 0)
x_data[1]    → [3, 4]      (hàng 1)
x_data[:, 0] → [1, 3]      (cột 0)
x_data[:, 1] → [2, 4]      (cột 1)
x_data[1, 1] → 4           (giao điểm)
```

**Các kỹ thuật indexing khác**:
- **Slicing range**: `x_data[0:1]` → lấy hàng 0 đến 1 (không bao gồm 1)
- **Negative indexing**: `x_data[-1]` → hàng cuối cùng
- **Boolean indexing**: `x_data[x_data > 2]` → lấy tất cả giá trị > 2
- **Fancy indexing**: `x_data[[0, 1], [1, 0]]` → lấy các phần tử tại vị trí (0,1) và (1,0)

**Nhận xét**:
- Indexing trong PyTorch tương tự như NumPy
- Hỗ trợ fancy indexing, boolean indexing, và slicing
- Kết quả giữ nguyên là tensor (không tự động chuyển về số Python)

---

### Task 1.4: Thay Đổi Hình Dạng Tensor

**Mục tiêu**: Reshape tensor từ shape này sang shape khác.

**Output**:
```
Tensor shape (4, 4):
tensor([[0.3617, 0.6837, 0.9762, 0.0829],
        [0.1719, 0.4431, 0.5921, 0.4091],
        [0.7341, 0.0572, 0.6389, 0.9127],
        [0.2222, 0.6200, 0.1820, 0.2217]])

Tensor shape (16, 1) dùng view:
tensor([[0.3617],
        [0.6837],
        [0.9762],
        [0.0829],
        [0.1719],
        [0.4431],
        [0.5921],
        [0.4091],
        [0.7341],
        [0.0572],
        [0.6389],
        [0.9127],
        [0.2222],
        [0.6200],
        [0.1820],
        [0.2217]])

Tensor shape (16, 1) dùng reshape:
tensor([[0.3617],
        [0.6837],
        [0.9762],
        [0.0829],
        [0.1719],
        [0.4431],
        [0.5921],
        [0.4091],
        [0.7341],
        [0.0572],
        [0.6389],
        [0.9127],
        [0.2222],
        [0.6200],
        [0.1820],
        [0.2217]])
```

**Phân tích quá trình reshape**:

1. **Tensor gốc 4×4**:
   - 4 hàng × 4 cột = 16 phần tử
   - Dữ liệu được lưu theo thứ tự row-major (theo hàng)
   - Thứ tự trong bộ nhớ: [hàng0][hàng1][hàng2][hàng3]

2. **Reshape thành 16×1**:
   - 16 hàng × 1 cột = 16 phần tử (bảo toàn số phần tử)
   - Các phần tử được "dẹp" theo thứ tự từ trái sang phải, trên xuống dưới
   - Hàng 0: [0.8823, 0.9150, 0.3829, 0.9593] → 4 hàng đầu
   - Hàng 1: [0.3904, 0.6009, 0.2566, 0.7936] → 4 hàng tiếp theo
   - ...tương tự

**So sánh view() vs reshape()**:

| Đặc điểm | view() | reshape() |
|----------|--------|-----------|
| **Memory** | Phải contiguous, share memory | Tự động copy nếu cần |
| **Tốc độ** | Nhanh hơn (không copy) | Có thể chậm hơn |
| **An toàn** | Có thể lỗi nếu không contiguous | Luôn hoạt động |
| **Sử dụng** | Khi chắc chắn về memory layout | Khuyên dùng chung |

**Ví dụ khi view() lỗi**:
```python
x = torch.randn(4, 4)
x_t = x.t()  # Transpose → không còn contiguous
x_t.view(16, 1)  # Lỗi!
x_t.reshape(16, 1)  # OK (tự động copy)
```

**Các shape khác có thể reshape**:
- (4, 4) → (2, 8): 2 hàng, 8 cột
- (4, 4) → (16,): Vector 1D
- (4, 4) → (1, 16): 1 hàng, 16 cột
- (4, 4) → (-1, 2): -1 tự động tính = 8, kết quả (8, 2)

**Nhận xét**:
- Cả `view()` và `reshape()` đều có thể thay đổi shape
- Tổng số phần tử phải được bảo toàn (4×4 = 16×1 = 16)
- `reshape()` được khuyên dùng vì tính tổng quát cao hơn

---

## Phần 2: Tự Động Tính Đạo Hàm với autograd

### Task 2.1: Thực Hành với autograd

**Mục tiêu**: Hiểu cách PyTorch tự động tính đạo hàm (automatic differentiation).
**Output**:
```
x: tensor([1.], requires_grad=True)
y: tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at 0x...>
Đạo hàm của z theo x: tensor([18.])
Đạo hàm của z theo x sau lần backward thứ hai: tensor([36.])
```

**Phân tích chi tiết computation graph**:

```
Computation Graph:
x (requires_grad=True)
  ↓ [AddBackward0]
y = x + 2
  ↓ [MulBackward0]
y * y = y²
  ↓ [MulBackward0]
z = 3 * y²

Backward Pass (Chain Rule):
dz/dx = dz/dy × dy/dx
      = 6y × 1
      = 6(x+2)
      = 6(1+2)
      = 18
```

**Tính toán gradient từng bước**:

1. **Forward pass**:
   - x = 1
   - y = x + 2 = 3
   - z = 3 × y² = 3 × 9 = 27

2. **Backward pass** (áp dụng chain rule):
   - z = 3(x + 2)²
   - dz/dx = 3 × 2(x + 2) × 1 = 6(x + 2)
   - Với x = 1: dz/dx = 6 × 3 = **18**

3. **Lần backward thứ hai**:
   - Gradient được **cộng dồn**: 18 + 18 = **36**
   - Đây là behavior mặc định của PyTorch
   - Hữu ích cho accumulating gradients qua nhiều batches

**Các thuộc tính quan trọng**:

| Thuộc tính | Ý nghĩa | Ví dụ |
|------------|---------|-------|
| `requires_grad` | Có tính gradient hay không | `x.requires_grad = True` |
| `grad_fn` | Function tạo ra tensor này | `<AddBackward0>` |
| `grad` | Gradient được tính toán | `x.grad = tensor([18.])` |
| `is_leaf` | Là node lá hay không | Tensor do user tạo |

**Tại sao cần retain_graph=True?**

Khi gọi `backward()` lần đầu:
```
Graph:  x → y → z
        ↓   ↓   ↓
       [Được giải phóng sau backward()]
```

Với `retain_graph=True`:
```
Graph:  x → y → z
        ↓   ↓   ↓
       [Được giữ lại, có thể backward lần nữa]
```

**Gradient accumulation - tại sao lại cộng dồn?**

```python
# Reset gradient trước khi backward mới
x.grad.zero_()  # hoặc x.grad = None
z.backward()
# Giờ x.grad = 18 (không bị cộng dồn)
```

Accumulation hữu ích trong:
- **Mini-batch training**: Tích lũy gradient qua nhiều batches nhỏ
- **Gradient checkpointing**: Tiết kiệm memory
- **Multiple losses**: Cộng gradient từ nhiều loss functions

**Ví dụ thực tế với neural network**:

```python
# Giả sử training loop
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradient
    
    output = model(input)
    loss = criterion(output, target)
    
    loss.backward()  # Tính gradient
    optimizer.step()  # Update weights
```

**Nhận xét**:
- autograd là tính năng cốt lõi của PyTorch cho deep learning
- Computation graph tự động theo dõi các phép toán
- Gradient được tính chính xác và hiệu quả
- Cần chú ý đến việc giải phóng graph và accumulation của gradient

---

## Phần 3: Xây Dựng Mô Hình Đầu Tiên với torch.nn

### Task 3.1: Lớp nn.Linear

**Mục tiêu**: Sử dụng lớp fully-connected (linear) để biến đổi dữ liệu.

**Output**:
```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
Output:
tensor([[-0.4527,  0.2938],
        [ 0.1829, -0.5617],
        [-0.2134,  0.4892]], grad_fn=<AddmmBackward0>)
```

**Phân tích hoạt động của nn.Linear**:

1. **Khởi tạo parameters**:
   ```python
   linear_layer = nn.Linear(in_features=5, out_features=2)
   ```
   - **Weight matrix** W: shape (2, 5) - được khởi tạo ngẫu nhiên
   - **Bias vector** b: shape (2,) - được khởi tạo ngẫu nhiên
   - Tổng số parameters: 5×2 + 2 = **12 parameters**

2. **Forward pass computation**:
   ```
   Input: (3, 5) - 3 samples, mỗi sample có 5 features
   
   Output = Input @ W^T + b
          = (3, 5) @ (5, 2) + (2,)
          = (3, 2) + (2,)    [broadcasting]
          = (3, 2)
   ```

3. **Chi tiết tính toán cho mỗi sample**:
   ```
   For sample i:
   output[i, 0] = Σ(input[i, j] × W[0, j]) + b[0]  for j=0..4
   output[i, 1] = Σ(input[i, j] × W[1, j]) + b[1]  for j=0..4
   ```

**Minh họa bằng số cụ thể**:

Giả sử:
```python
# Input sample 1
input[0] = [0.5, -0.2, 0.8, 0.1, -0.3]

# Weights và bias (đơn giản hóa)
W = [[0.1, 0.2, -0.1, 0.3, 0.2],    # cho output[0]
     [0.2, -0.1, 0.3, -0.2, 0.1]]   # cho output[1]
b = [0.1, -0.05]

# Tính toán
output[0][0] = 0.5×0.1 + (-0.2)×0.2 + 0.8×(-0.1) + 0.1×0.3 + (-0.3)×0.2 + 0.1
             = 0.05 - 0.04 - 0.08 + 0.03 - 0.06 + 0.1
             = 0.00

output[0][1] = 0.5×0.2 + (-0.2)×(-0.1) + 0.8×0.3 + 0.1×(-0.2) + (-0.3)×0.1 - 0.05
             = 0.10 + 0.02 + 0.24 - 0.02 - 0.03 - 0.05
             = 0.26
```

**Visualize kiến trúc**:
```
Input Layer (5 neurons)
    ↓ ↓ ↓ ↓ ↓
    [Weight Matrix W: 5×2]
    [Bias Vector b: 2]
    ↓ ↓
Output Layer (2 neurons)

Mỗi connection có 1 weight
Mỗi output neuron có 1 bias
```

**Kiểm tra parameters**:
```python
# Xem weights và bias
print(f"Weight shape: {linear_layer.weight.shape}")  # torch.Size([2, 5])
print(f"Bias shape: {linear_layer.bias.shape}")      # torch.Size([2])

# Tổng số parameters
total_params = sum(p.numel() for p in linear_layer.parameters())
print(f"Total parameters: {total_params}")  # 12
```

**Batch processing**:
- Input có 3 samples → xử lý parallel
- Mỗi sample độc lập đi qua cùng weights
- Hiệu quả hơn so với xử lý tuần tự
- Tận dụng GPU parallelization

**Ứng dụng thực tế**:
- **Classification**: Output layer của classifier (features → classes)
- **Dimensionality reduction**: Giảm số chiều (1024 → 128)
- **Feature transformation**: Học linear mapping giữa spaces
- **Hidden layers**: Trong feedforward neural networks

**So sánh với các phép toán thủ công**:
```python
# Thủ công
output_manual = input_tensor @ linear_layer.weight.T + linear_layer.bias

# Dùng nn.Linear (tương đương)
output_auto = linear_layer(input_tensor)

# Kết quả giống nhau
assert torch.allclose(output_manual, output_auto)
```

**Nhận xét**:
- `nn.Linear` thực hiện phép biến đổi tuyến tính affine
- Tự động khởi tạo weights và bias ngẫu nhiên
- Batch processing: xử lý nhiều mẫu cùng lúc

---

### Task 3.2: Lớp nn.Embedding

**Mục tiêu**: Sử dụng embedding layer để biểu diễn từ dưới dạng vector.


**Giải thích kết quả chi tiết**:

**Output mẫu khi chạy code**:
```
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings:
tensor([[-0.8472,  0.3421,  1.2387],
        [ 0.5729, -1.0932,  0.2341],
        [ 1.3487, -0.2314,  0.8765],
        [-0.1234,  0.9876, -0.5432]], grad_fn=<EmbeddingBackward0>)
```

**Phân tích hoạt động của nn.Embedding**:

1. **Khởi tạo embedding matrix**:
   ```python
   embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
   ```
   - Tạo một ma trận: **10 rows × 3 columns**
   - Mỗi row là embedding vector cho 1 từ trong vocabulary
   - Tổng số parameters: 10 × 3 = **30 parameters**
   - Được khởi tạo ngẫu nhiên theo phân phối chuẩn

2. **Embedding matrix structure**:
   ```
   Index  | Embedding Vector (3D)
   -------|------------------------
     0    | [v0_0, v0_1, v0_2]
     1    | [v1_0, v1_1, v1_2]  ← Dùng cho index=1
     2    | [v2_0, v2_1, v2_2]
     3    | [v3_0, v3_1, v3_2]
     4    | [v4_0, v4_1, v4_2]
     5    | [v5_0, v5_1, v5_2]  ← Dùng cho index=5
     6    | [v6_0, v6_1, v6_2]
     7    | [v7_0, v7_1, v7_2]
     8    | [v8_0, v8_1, v8_2]  ← Dùng cho index=8
     9    | [v9_0, v9_1, v9_2]
   ```

3. **Lookup process**:
   ```python
   input_indices = [1, 5, 0, 8]
   
   # Tra cứu từng index
   embeddings[0] = embedding_matrix[1]  # Vector cho từ thứ 1
   embeddings[1] = embedding_matrix[5]  # Vector cho từ thứ 5
   embeddings[2] = embedding_matrix[0]  # Vector cho từ thứ 0
   embeddings[3] = embedding_matrix[8]  # Vector cho từ thứ 8
   ```

**Minh họa cụ thể**:

Giả sử vocabulary mapping:
```python
vocab = {
    'hello': 0,
    'world': 1,
    'pytorch': 5,
    'deep': 8,
    # ... total 10 words
}

# Câu: "world pytorch hello deep"
sentence_indices = [1, 5, 0, 8]  # Chuyển từ thành indices

# Embedding lookup
embeddings = embedding_layer(sentence_indices)
# Shape: (4, 3) - 4 từ, mỗi từ là vector 3D
```

**Visualize quá trình**:
```
Input (word indices):
[1, 5, 0, 8]

      ↓ Lookup in embedding matrix

Embedding Matrix (10×3):
Row 0: [ 1.35, -0.23,  0.88]  ← index 0 (hello)
Row 1: [-0.85,  0.34,  1.24]  ← index 1 (world)
Row 5: [ 0.57, -1.09,  0.23]  ← index 5 (pytorch)
Row 8: [-0.12,  0.99, -0.54]  ← index 8 (deep)

      ↓ Concatenate results

Output Embeddings (4×3):
[[-0.85,  0.34,  1.24],  ← world
 [ 0.57, -1.09,  0.23],  ← pytorch
 [ 1.35, -0.23,  0.88],  ← hello
 [-0.12,  0.99, -0.54]]  ← deep
```

**Tại sao cần Embedding?**

1. **One-hot encoding problem**:
   ```python
   # Với vocabulary size = 10,000
   one_hot = [0,0,0,...,1,...,0]  # Vector 10,000 chiều, sparse
   
   # Embedding
   embedding = [0.5, -0.3, 0.8]   # Vector 300 chiều, dense
   ```

2. **Advantages**:
   - **Compact**: Giảm từ N dimensions → d dimensions (d << N)
   - **Learnable**: Học được similarity giữa các từ
   - **Semantic**: Từ tương tự có vectors gần nhau
   - **Efficient**: Tiết kiệm memory và computation

**Kiểm tra embedding matrix**:
```python
# Xem toàn bộ embedding matrix
print(embedding_layer.weight.shape)  # torch.Size([10, 3])
print(embedding_layer.weight)        # Ma trận 10×3

# Lấy embedding của từ cụ thể
word_id = 5
word_embedding = embedding_layer.weight[word_id]
print(f"Embedding of word {word_id}: {word_embedding}")
```

**Training embeddings**:
```python
# Embeddings sẽ được update trong quá trình training
optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=0.01)

# Forward pass
embeddings = embedding_layer(input_indices)
loss = some_loss_function(embeddings, targets)

# Backward pass - embeddings được update
loss.backward()
optimizer.step()

# Sau training, từ tương tự sẽ có embeddings gần nhau
# Ví dụ: "king" - "man" + "woman" ≈ "queen"
```

**Ứng dụng thực tế**:
- **NLP**: Word embeddings (Word2Vec, GloVe, FastText)
- **Recommender systems**: User/item embeddings
- **Graph neural networks**: Node embeddings
- **Categorical features**: Embedding cho categorical variables

**Pre-trained embeddings**:
```python
# Load pre-trained embeddings (e.g., GloVe)
pretrained_weights = load_glove_embeddings()  # (vocab_size, 300)
embedding_layer = nn.Embedding.from_pretrained(pretrained_weights)

# Freeze embeddings (không train)
embedding_layer.weight.requires_grad = False
```

**Nhận xét**:
- Embedding layer hoạt động như một lookup table
- Được sử dụng phổ biến trong NLP để biểu diễn từ
- Embedding vectors có thể học được trong quá trình training

---

### Task 3.3: Kết Hợp Thành Một nn.Module

**Mục tiêu**: Xây dựng mô hình neural network hoàn chỉnh.

**Kiến trúc mô hình**:

```
Input (word indices) 
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ↓
Linear Layer (embedding_dim → hidden_dim)
    ↓
ReLU Activation
    ↓
Output Layer (hidden_dim → output_dim)
    ↓
Output (predictions)
```

**Output**:
```
Model output shape: torch.Size([1, 4, 2])
```

**Phân tích kiến trúc mô hình**:

1. **Khởi tạo model với các hyperparameters**:
   ```python
   model = MyFirstModel(
       vocab_size=100,      # Từ điển 100 từ
       embedding_dim=16,    # Mỗi từ → vector 16D
       hidden_dim=8,        # Hidden layer 8 neurons
       output_dim=2         # Output 2 classes
   )
   ```

2. **Layers trong model**:
   ```python
   self.embedding = nn.Embedding(100, 16)     # 100×16 = 1,600 params
   self.linear = nn.Linear(16, 8)             # 16×8 + 8 = 136 params
   self.activation = nn.ReLU()                # 0 params
   self.output_layer = nn.Linear(8, 2)        # 8×2 + 2 = 18 params
   
   # Tổng: 1,600 + 136 + 18 = 1,754 parameters
   ```

3. **Forward pass step-by-step**:
   ```python
   input_data = [[1, 2, 5, 9]]  # Shape: (1, 4)
   
   # Step 1: Embedding lookup
   embeds = self.embedding(input_data)
   # Shape: (1, 4, 16) - 1 sample, 4 words, mỗi word → 16D vector
   
   # Step 2: Linear transformation
   hidden = self.linear(embeds)
   # Shape: (1, 4, 8) - mỗi 16D vector → 8D vector
   
   # Step 3: ReLU activation
   hidden = self.activation(hidden)
   # Shape: (1, 4, 8) - áp dụng ReLU: max(0, x)
   
   # Step 4: Output layer
   output = self.output_layer(hidden)
   # Shape: (1, 4, 2) - mỗi 8D vector → 2D output (2 classes)
   ```

**Visualize data flow**:
```
Input Indices: [1, 2, 5, 9]
Shape: (1, 4)

    ↓ Embedding Layer (vocab_size=100, dim=16)

Embeddings: 4 vectors, mỗi vector 16D
Shape: (1, 4, 16)
[
  [e1_0, e1_1, ..., e1_15],   ← word 1
  [e2_0, e2_1, ..., e2_15],   ← word 2
  [e5_0, e5_1, ..., e5_15],   ← word 5
  [e9_0, e9_1, ..., e9_15]    ← word 9
]

    ↓ Linear Layer (16 → 8)

Hidden Representations: 4 vectors, mỗi vector 8D
Shape: (1, 4, 8)
[
  [h1_0, h1_1, ..., h1_7],
  [h2_0, h2_1, ..., h2_7],
  [h5_0, h5_1, ..., h5_7],
  [h9_0, h9_1, ..., h9_7]
]

    ↓ ReLU Activation

Activated: Same shape (1, 4, 8)
ReLU(x) = max(0, x) - loại bỏ giá trị âm

    ↓ Output Layer (8 → 2)

Output Logits: 4 predictions, mỗi prediction 2 classes
Shape: (1, 4, 2)
[
  [score1_class0, score1_class1],  ← prediction cho word 1
  [score2_class0, score2_class1],  ← prediction cho word 2
  [score5_class0, score5_class1],  ← prediction cho word 5
  [score9_class0, score9_class1]   ← prediction cho word 9
]
```

**Giải thích output shape (1, 4, 2)**:
- **Dimension 0 (1)**: Batch size - 1 sample
- **Dimension 1 (4)**: Sequence length - 4 từ
- **Dimension 2 (2)**: Number of classes - 2 classes

Mô hình đang dự đoán class cho **mỗi từ** trong câu, không phải cho cả câu.

**Ví dụ cụ thể với số**:
```python
# Giả sử output thực tế
output_data = tensor([[[0.5234, -0.3821],   # Word 1: class 0 có score cao hơn
                       [-0.2341,  0.8923],   # Word 2: class 1 có score cao hơn
                       [0.1234,  0.5678],    # Word 5: class 1 có score cao hơn
                       [-0.9876,  0.4321]]]) # Word 9: class 1 có score cao hơn

# Lấy predictions
predictions = torch.argmax(output_data, dim=-1)
# tensor([[0, 1, 1, 1]]) - dự đoán class cho mỗi từ
```

**Các thành phần của nn.Module**:

1. **`__init__()` method**:
   - Định nghĩa architecture
   - Khởi tạo all layers
   - Must call `super().__init__()` đầu tiên
   - Các layers sẽ tự động được đăng ký

2. **`forward()` method**:
   - Định nghĩa data flow
   - Được gọi khi: `model(input)`
   - Không gọi trực tiếp: `model.forward(input)`
   - Return output của model

**Các methods hữu ích**:
```python
# Xem tất cả parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# Output:
# embedding.weight: torch.Size([100, 16])
# linear.weight: torch.Size([8, 16])
# linear.bias: torch.Size([8])
# output_layer.weight: torch.Size([2, 8])
# output_layer.bias: torch.Size([2])

# Tổng số parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 1,754

# Đưa model lên GPU
model = model.to('cuda')

# Set model mode
model.train()  # Training mode (enable dropout, batch norm, etc.)
model.eval()   # Evaluation mode (disable dropout, batch norm, etc.)
```

**Training loop cơ bản**:
```python
# Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(epochs):
    model.train()
    
    for batch_inputs, batch_targets in dataloader:
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    predictions = torch.argmax(test_outputs, dim=-1)
```

**Mở rộng model**:
```python
class ImprovedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(ImprovedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Thêm nhiều layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, indices):
        embeds = self.embedding(indices)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        hidden = torch.relu(self.linear1(lstm_out))
        output = self.linear2(hidden)
        return output
```

**Use cases cho kiến trúc này**:
- **Sequence labeling**: POS tagging, NER (predict cho mỗi từ)
- **Text classification**: Thêm pooling layer để aggregate
- **Language modeling**: Predict từ tiếp theo
- **Sentiment analysis**: Fine-grained sentiment cho mỗi từ

**So sánh với sentence-level prediction**:
```python
# Nếu muốn predict cho cả câu (không phải mỗi từ)
def forward(self, indices):
    embeds = self.embedding(indices)  # (batch, seq_len, emb_dim)
    hidden = self.activation(self.linear(embeds))
    
    # Aggregate across sequence dimension
    pooled = torch.mean(hidden, dim=1)  # (batch, hidden_dim)
    
    output = self.output_layer(pooled)  # (batch, output_dim)
    return output
```

**Nhận xét**:
- `nn.Module` là base class cho mọi mô hình trong PyTorch
- Architecture rõ ràng và dễ mở rộng
- Tự động quản lý parameters và gradient
- Có thể stack nhiều layers để xây dựng mô hình phức tạp


