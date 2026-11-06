# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603516
- **Họ và tên:** Trần Quang Nam
- **SEED:** 8778

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Dưới đây là bảng trích xuất số liệu thực tế đo lường trên tập dữ liệu riêng (SEED = 8778) từ Phần 5 và `lab1_report.json`:

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Throughput (tin/s) | Latency p50 (ms) | Latency p95 (ms) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.792 | 0.884 | 0.889 | 5.075 | 15,750 | 0.045 | 0.080 | $0.000 |
| **2. BoW + MLP (Deep Learning)** | 0.989 | 0.906 | 0.946 | 0.919 | 9.925 | 20,964 | 0.263 | 0.327 | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.882 | 0.938 | 0.909 | 0.974 | 2.313 | 119 | 16.863 | 32.422 | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.967 | 0.906 | 0.935 | 0.919 | 2.730 | 1,267 | 6.764 | 10.077 | $0.000 |
| **3c. Encoder (MiniLM) + MLP head**| 0.956 | 0.896 | 0.925 | 0.947 | 3.354 | 1,723 | 6.670 | 8.787 | $0.000 |
| **4a. LLM nhỏ zero-shot (`gpt-6-luna`)** | 0.952 | 1.000 | 0.976 | 0.976 | 0.000 | 0.75 | 1,010.23 | 3,956.81 | $0.008 |
| **4b. LLM lớn zero-shot (`gpt-6.1-terra`)** | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.92 | 1,092.93 | 1,298.60 | $0.078 |
| **4c. LLM nhỏ few-shot (`gpt-6-luna`)** | 1.000 | 0.950 | 0.974 | 0.974 | 0.000 | 1.00 | 920.30 | 1,390.21 | $0.019 |

---

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất so với kết quả thực nghiệm:

1. **Chi phí chênh lệch giữa LLM lớn và LLM nhỏ (`llm_models`):**
   - *Dự đoán:* Model lớn có F1 cao hơn model nhỏ khoảng $0.02 - 0.05$ và giá đắt gấp $\sim 2.5$ lần.
   - *Thực tế:* Model lớn đạt $F_1 = 1.000$, model nhỏ đạt $F_1 = 0.976$ (độ chênh lệch $0.024$ khớp dự đoán). Tuy nhiên, chi phí thực tế của model lớn là $\$0.0775$/1k tin, đắt gấp **10 lần** so với model nhỏ ($\$0.00775$/1k tin) chứ không phải $2.5$ lần.
   - *Nguyên nhân:* Dự đoán dựa trên cảm tính thông thường về chênh lệch giá, trong khi cơ cấu định giá API giữa dòng frontier/flagship (`gpt-6.1-terra`) và dòng mini/distilled (`gpt-6-luna`) chênh lệch thực tế tới $10\times$ ($1.00$ USD vs $0.10$ USD / 1M input tokens).

2. **Hiệu quả của Prompt Few-shot (`llm_fewshot`):**
   - *Dự đoán:* Few-shot sẽ giúp tăng $F_1$ nhẹ ($\sim 0.02$), nhưng chi phí/tin tăng $\sim 4$ lần do prompt dài hơn.
   - *Thực tế:* $F_1$ không hề tăng mà giảm nhẹ từ $0.976$ (zero-shot) xuống $0.974$ (few-shot); chi phí thực tế chỉ tăng $\sim 2.4$ lần ($\$0.00031 \rightarrow \$0.00075$ cho 40 tin).
   - *Nguyên nhân:* Nhiệm vụ nhị phân (spam vs ham) là một bài toán phân loại đơn giản, LLM đã nắm rõ bản chất ngữ nghĩa ngay từ prompt mô tả. Việc chèn thêm 4 ví dụ không cung cấp thêm tri thức biên mà lại vô tình đưa thêm một ít nhiễu về độ dài và phong cách, dẫn đến sụt giảm nhỏ về recall (từ $1.000$ xuống $0.950$).

3. **Hiện tượng Learning Rate lớn ở Logistic Regression (`ml_lr`):**
   - *Dự đoán:* Với `lr=50`, hàm mất mát sẽ không giảm đều mà tăng vọt hoặc dao động liên tục quanh đáy suốt quá trình train.
   - *Thực tế:* Ở vài epoch đầu tiên, loss nhảy vọt đột ngột lên mức $\sim 2.6$ (bước nhảy gradient quá lớn nhảy vượt điểm cực tiểu), nhưng sau đó đường loss vẫn hội tụ ổn định dần về mức thấp thay vì dao động hỗn loạn liên tục cho đến epoch 300.
   - *Nguyên nhân:* Do mặt cong của hàm mục tiêu Binary Cross-Entropy kết hợp cùng đặc tính co biên của hàm Sigmoid đã kìm hãm độ lớn của gradient khi $z$ tiến ra vùng bão hòa, giúp mô hình tự "hãm phanh" và tiến dần về cực trị sau cú vọt ban đầu.

---

## 4. Phân tích 50 case khó (Stress Test)

### 4.1. Tổng quan độ sụt giảm hiệu năng

| Phương pháp | $F_1$ Test thường (SUB) | $F_1$ 50 case khó | Accuracy 50 case khó |
| :--- | :---: | :---: | :---: |
| **1. BoW + LogReg** | 0.889 | **0.333** | 52% |
| **2. BoW + MLP** | 0.919 | **0.450** | 56% |
| **3a. Emb + kNN** | 0.974 | 0.750 | 72% |
| **3b. Emb + LogReg** | 0.919 | 0.612 | 62% |
| **3c. Emb + MLP head** | 0.947 | 0.640 | 64% |
| **4a. LLM nhỏ zero-shot** | 0.976 | **1.000** | 100% |
| **4d. LLM nhỏ + prompt phòng thủ** | 0.976 | **1.000** | 100% |

- **Mô hình tụt mạnh nhất:** `BoW + LogReg` bị sụt giảm thảm hại nhất, $F_1$ rơi từ $0.889$ xuống còn $0.333$ (accuracy chỉ còn $52\%$, tương đương đoán mò ngẫu nhiên).
- **Nhóm case gây khó khăn nhất cho BoW:**
  - *Spam giọng tự nhiên (natural_spam):* Độ chính xác chỉ đạt $10\%$.
  - *Thao túng / lừa đảo (manipulation):* Độ chính xác chỉ đạt $12\%$.
  - *Né từ khóa (obfuscation):* Độ chính xác chỉ đạt $17\%$.

### 4.2. Cơ chế lỗi qua 2 case cụ thể

1. **Case #21 (Thao túng / Mạo danh người thân):**
   - *Nội dung:* `"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"` (Nhãn thật: `spam`).
   - *Kết quả BoW:* Dự đoán nhầm thành `ham`.
   - *Cơ chế sai:* Mô hình BoW chỉ phân loại dựa trên sự xuất hiện độc lập của các từ khóa bề mặt. Câu này sử dụng toàn các từ đời thường, mang tính gia đình và tình cảm (`"Mum"`, `"phone"`, `"friend"`, `"explain"`). Những từ này trong tập train có trọng số $w$ âm lớn (kéo mạnh về phía `ham`). BoW hoàn toàn không có khả năng hiểu ngữ nghĩa ngữ cảnh (bối cảnh lừa đảo "Hi Mum" tinh vi để chuyển tiền khẩn cấp).

2. **Case #30 (Né từ khóa bằng cách giãn ký tự):**
   - *Nội dung:* `"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"` (Nhãn thật: `spam`).
   - *Kết quả BoW:* Dự đoán nhầm thành `ham`.
   - *Cơ chế sai:* Bộ tokenizer của mô hình tách từ theo biểu thức chính quy `[a-z0-9£$€]+`. Do kẻ lừa đảo chèn khoảng trắng giữa các chữ cái, từ `"FREE"` bị bẻ gãy thành các token đơn lẻ `["f", "r", "e", "e"]`. Các chữ cái đơn lẻ này không nằm trong từ điển vocab $2000$ từ (hoặc mang trọng số bằng $0$), khiến vector BoW trở thành vector toàn $0$, làm tổng trọng số nghiêng hoàn toàn về giá trị bias $b$ âm (đoán `ham`).

---

## 5. Thử thách cá nhân

- **Đề bài được giao (Đề #1, do `SEED % 6 = 8778 % 6 = 0`):**
  > *“ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?”*

- **Cách thực hiện:**
  Tính ma trận xác suất dự đoán $p = \sigma(X_{\text{te}} \cdot w + b)$ trên tập test từ mô hình Logistic Regression tự code ở Phần 1, sau đó áp dụng lần lượt các ngưỡng phân loại $\tau \in \{0.2, 0.5, 0.8\}$: nếu $p \ge \tau$ thì gán nhãn $1$ (spam), ngược lại gán nhãn $0$ (ham). Tính toán lại `metrics(y_te, pred)`.

- **Kết quả thực nghiệm:**

| Ngưỡng ($\tau$) | Precision | Recall | $F_1$ Score |
| :---: | :---: | :---: | :---: |
| **0.2** | 0.869 | **0.896** | 0.882 |
| **0.5** | **1.000** | 0.792 | **0.884** |
| **0.8** | **1.000** | 0.573 | 0.728 |

- **Giải thích & Tư vấn cho ngân hàng:**
  - Khi giảm ngưỡng xuống $0.2$, mô hình trở nên nhạy cảm hơn với spam, giúp tăng Recall từ $0.792$ lên $0.896$ nhưng Precision bị tụt xuống $0.869$ (bắt đầu có nhiều tin hợp lệ bị chặn nhầm - False Positive tăng).
  - Khi tăng ngưỡng lên $0.8$, mô hình chỉ kết luận spam khi xác suất cực kỳ chắc chắn, Precision đạt tuyệt đối $1.000$ nhưng Recall rớt sâu xuống $0.573$ (bỏ lọt hơn $42\%$ tin rác).
  - **Lựa chọn cho ngân hàng:** Ngân hàng bắt buộc phải **chọn ngưỡng cao (0.8 hoặc thậm chí cao hơn)**. Đối với nghiệp vụ ngân hàng, chi phí của việc chặn nhầm một tin nhắn mã xác thực giao dịch OTP (False Positive) là vô cùng nghiêm trọng: làm đứt gãy trải nghiệm khách hàng, gây bức xúc, hoặc đình trệ các giao dịch tài chính quan trọng. Việc để lọt một vài tin nhắn rác/tiếp thị (False Negative) dễ được người dùng bỏ qua hơn nhiều so với việc không nhận được mã OTP chuyển tiền.

---

## 6. Phương án cho sản phẩm thật

### 6.1. Bối cảnh & Yêu cầu
- **Hạ tầng:** 10 triệu SMS/ngày (lưu lượng trung bình $\sim 116$ tin/giây, giờ cao điểm có thể lên tới $500 - 1,000$ tin/giây).
- **Ràng buộc SLA:** Thời gian phản hồi $p_{95} < 50\text{ ms/tin}$.

### 6.2. Đánh giá các phương án dựa trên số liệu thực nghiệm
1. **Các phương án LLM API (4a, 4b, 4c):**
   - *Loại bỏ ngay lập tức:* Độ trễ $p_{95}$ dao động từ $1,298\text{ ms}$ đến $3,956\text{ ms}$ (vi phạm SLA gấp $25 - 80$ lần). Đồng thời, thông lượng tuần tự chỉ đạt $\approx 0.75 - 1.0\text{ tin/s}$, không thể tải nổi mức $116\text{ tin/s}$.
   - *Chi phí:* Lọc 10 triệu tin/ngày tương đương $300\text{ triệu tin/tháng}$. Nếu dùng `gpt-6-luna` tốn khoảng $\$2,325$/tháng, dùng `gpt-6.1-terra` tốn tới hơn $\$23,250$/tháng — quá lãng phí cho bài toán lọc SMS mức nhà mạng.
2. **Phương án BoW (1, 2):**
   - Siêu nhanh ($p_{95} \le 0.33\text{ ms}$, thông lượng $> 15,000\text{ tin/s}$), hoàn toàn miễn phí vận hành inference. Tuy nhiên, khả năng chống đỡ trước các tin nhắn ngụy trang, né từ khóa hoặc spam ngữ cảnh mới rất kém (chỉ đạt $52 - 56\%$ accuracy ở Stress Test).
3. **Phương án Embedding MiniLM (3a, 3b, 3c):**
   - Đạt SLA $p_{95} < 50\text{ ms}$: `3c. Encoder + MLP head` đạt $p_{95} = 8.79\text{ ms}$; `3b. Encoder + LogReg` đạt $p_{95} = 10.08\text{ ms}$; `3a. kNN` đạt $p_{95} = 32.42\text{ ms}$.
   - Khả năng tổng quát hóa ngữ nghĩa vượt trội so với BoW, chi phí API bằng $0$ (chạy on-premise).

### 6.3. Đề xuất kiến trúc: Thiết kế lai (Two-Stage Router)
Để vừa đảm bảo SLA khắt khe của viễn thông, vừa đạt độ chính xác tối ưu trước các thủ đoạn lừa đảo tinh vi, tôi đề xuất kiến trúc **Router 2 tầng**:
- **Tầng 1 (Fast Filter - Local Model):** Sử dụng mô hình `Encoder (all-MiniLM-L6-v2) + MLP Head` triển khai trên máy chủ cục bộ (CPU/GPU nhẹ).
  - Nếu $p(\text{spam}) < 0.20 \rightarrow$ Phê duyệt ngay lập tức là `ham` (an toàn cho OTP, tin nhắn công việc).
  - Nếu $p(\text{spam}) > 0.85 \rightarrow$ Chặn ngay lập tức là `spam`.
  - Tầng 1 giải quyết khoảng $95 - 98\%$ tổng lưu lượng với độ trễ chỉ $\sim 8.8\text{ ms}$, hoàn toàn đáp ứng SLA $< 50\text{ ms}$.
- **Tầng 2 (Deep Filter - LLM API Fallback):** Chỉ những tin nhắn nằm trong vùng phân vân lưỡng lự $p \in [0.20, 0.85]$ (chiếm $\approx 2 - 5\%$ lưu lượng) mới được chuyển tiếp bất đồng bộ qua hàng đợi (queue) để `LLM nhỏ (`gpt-6-luna`) với Prompt phòng thủ` phân tích ngữ cảnh chuyên sâu.
- **Sự đánh đổi:** Chấp nhận đầu tư chi phí hạ tầng máy chủ nội bộ để chạy encoder MiniLM và một khoản chi phí API nhỏ ($\sim \$50 - \$100$/tháng cho các ca khó) đổi lấy một hệ thống lọc tin nhắn đạt chuẩn SLA công nghiệp, có khả năng miễn nhiễm cao với kỹ thuật lừa đảo mới.

---

## 7. AI log và điều đã học

### 7.1. Nhật ký sử dụng AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 0: Data Leakage** | "trùng tin nhắn trong tập train với test thì có sao k" | AI giải thích khái niệm data leakage: mô hình học thuộc tin nhắn trùng khiến điểm test cao ảo, không đo được khả năng tổng quát hóa thực tế. | Tự viết lại câu trả lời 0.1 bằng lời của mình, kiểm tra lại số lượng tin trùng lặp bằng hàm `duplicated()` trong notebook. |
| **Phần 1: Machine Learning (Lý thuyết)** | "đạo hàm logreg sao ra được p - y thế" | AI tóm tắt các bước đạo hàm hàm loss BCE theo trọng số $w$ qua quy tắc chuỗi (chain rule) với hàm sigmoid $\sigma(z)$. | Lấy giấy nháp tự tính lại từng bước đạo hàm, đối chiếu với công thức `(X.T @ error) / n` để tự gõ code vòng lặp `train_logreg`. |
| **Phần 1: Machine Learning (Lý thuyết)** | "đoán thiếu spam thì error âm hay dương " | AI giải thích nhãn thật $y = 1$, dự đoán $p < 0.5$ nên $error = p - y < 0$ (âm). Khi cập nhật $w = w - lr \times grad$, trừ số âm thành cộng nên trọng số tăng lên. | Kiểm tra lại logic câu hỏi giải thích 1 và thử in trọng số của từ 'free' sau vài epoch xem có tăng đúng như vậy không. |
| **Phần 1: Machine Learning (Lý thuyết)** | "lr to quá như 50 thì loss nó bị gì" | AI giải thích learning rate quá lớn làm bước nhảy vượt qua điểm cực tiểu, loss có thể vọt lên rất cao hoặc dao động không hội tụ. | Dùng làm cơ sở để điền dự đoán `predict("ml_lr", ...)` trước khi chạy thí nghiệm 1.B. |
| **Phần 2: Deep Learning (Khái niệm)** | "tại sao bắt buộc phải có relu giữa 2 lớp linear" | AI giải thích nếu không có hàm phi tuyến, tích hai ma trận trọng số $W_2(W_1x + b_1) + b_2$ suy biến thành một phép biến đổi tuyến tính đơn thuần, mạng nhiều lớp không khác gì 1 lớp. | Tự viết `predict("dl_relu", ...)` dự đoán F1 không đổi khi bỏ ReLU, sau đó kiểm chứng qua code Thí nghiệm 2.B ra đúng $F_1 \approx 0.946$. |
| **Phần 2: Deep Learning (Khái niệm)** | "quên opt.zero_grad() trong pytorch thì bị lỗi gì" | AI giải thích PyTorch mặc định cộng dồn gradient qua các epoch/batch; nếu không xóa, gradient cũ sẽ làm bước cập nhật đi sai hướng khiến loss không giảm đều. | Đối chiếu với 5 bước trong vòng lặp `train_mlp` và dùng ý này để viết câu trả lời cho mục "Giải thích 2". |
| **Phần 2: Deep Learning (Khái niệm)** | "dấu hiệu overfit trên biểu đồ loss là gì" | AI nêu dấu hiệu: train loss tiếp tục giảm sâu về 0 nhưng val loss bắt đầu đi ngang hoặc tăng ngược trở lại. | Quan sát biểu đồ Thí nghiệm 2.A, xác định val loss chạm đáy ở epoch ~20 để trả lời câu hỏi 2.A. |
| **Phần 3: Embedding (Bản chất)** | "cosine similarity khác dot product chỗ nào sao phải chuẩn hóa" | AI giải thích dot product phụ thuộc vào độ dài vector (câu dài lặp từ sẽ có điểm cao), còn cosine chia cho độ dài (L2 norm) để chỉ đo góc/hướng ngữ nghĩa. | Nhận diện lỗi quên chuẩn hóa trong phần "Bắt lỗi 2", tự viết hàm `top_k_similar` bằng phép nhân ma trận NumPy `@` mà không cần vòng lặp. |
| **Phần 3: Embedding (Bản chất)** | "knn có cần train k" | AI giải thích kNN là lazy learning, không có tham số để tối ưu hóa mà chỉ lưu sẵn vector train làm bộ nhớ, khi có query thì tính khoảng cách rồi bỏ phiếu. | Tự viết hàm `knn_predict` và trả lời câu hỏi checkpoint về việc kNN bị chậm khi tập dữ liệu lên tới hàng triệu tin. |
| **Phần 4: LLM API (Code & Prompt)** | "regex lọc chữ spam hoặc ham trong text" | AI đưa ra đoạn regex bắt `\b(spam|ham)\b`. | Thấy regex của AI còn sót trường hợp tin nhắn có "Not spam" (bị bắt nhầm thành spam). Đã viết lại hàm `re.search(r"\bnot (a )?spam\b")` xử lý trước, đồng thời trả về `None` nếu câu trả lời rỗng hoặc không rõ ràng. |
| **Phần 4: LLM API (Code & Prompt)** | "sao gọi openai gpt-6 báo lỗi max_tokens" | AI giải thích các dòng model mới của OpenAI dùng tham số `max_completion_tokens` thay cho `max_tokens` và cần tắt phần suy luận bằng `reasoning_effort="none"` nếu muốn trả lời ngắn gọn. | Kiểm tra lại cấu hình trong dictionary `PROVIDERS` và truyền `**provider_kwargs(max_tokens)` vào hàm `call_openai`. |
| **Phần 4: LLM API (Code & Prompt)** | "viết prompt phân loại tin nhắn ngắn gọn chỉ ra 1 chữ" | AI gợi ý mẫu system prompt và định dạng: `Message: """...""" \nLabel:`. | Chỉnh sửa lại hàm `build_classification_prompt` để hỗ trợ thêm cả chế độ few-shot (chèn danh sách examples) theo đúng yêu cầu TODO 4.2. |

### 7.2. Điều đã học được
1. **Accuracy trên tập test chuẩn rất dễ đánh lừa:** Một mô hình đạt accuracy $>98\%$ trên tập test chuẩn (cùng phân phối) vẫn có thể sụp đổ hoàn toàn (xuống còn $52\%$) khi đối mặt với dữ liệu ngoài thực tế bị cố tình ngụy trang, tách từ hoặc thay đổi ngữ cảnh.
2. **Kích thước mô hình và độ phức tạp không đồng nghĩa với hiệu quả:** Trên biểu diễn Bag-of-Words, Deep Learning (MLP) không mang lại đột phá đáng kể so với Logistic Regression đơn giản vì dữ liệu văn bản ngắn đã gần như phân tách tuyến tính được. Thêm vào đó, việc lạm dụng prompt Few-shot cho LLM có thể làm tăng chi phí token lên gấp nhiều lần mà hiệu quả thực tế lại sụt giảm do nhiễu ngữ cảnh.
3. **Độ trễ và bài toán đánh đổi trong sản xuất:** Trong triển khai hệ thống AI kỹ thuật thực tế, độ trễ và chi phí suy luận (inference cost/latency) là những rào cản mang tính quyết định không kém gì thước đo $F_1$. Mô hình mạnh nhất chưa chắc đã là mô hình được chọn nếu nó vi phạm SLA hệ thống.