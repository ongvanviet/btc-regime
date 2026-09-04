# btc-regime

Web app mã nguồn mở đọc **trạng thái rủi ro (risk regime) của Bitcoin** mỗi ngày,
lấy cảm hứng từ các mô tả công khai về sản phẩm cùng loại. Đây là bản tái tạo theo
mô tả, **không** dùng thuật toán gốc và **không** liên quan tới Glassnode/Swissblock.

Mỗi ngày mô hình xếp Bitcoin vào 1 trong 4 trạng thái:

| Mã | Trạng thái | Nhãn | Risk | Phân bổ BTC |
|---:|---|---|---|---:|
| 2 | Strong-On | Euphoria | Risk-On | 100% |
| 1 | Mild-On | Accumulation | Risk-On | 60% |
| −1 | Mild-Off | Caution | Risk-Off | 25% |
| −2 | Strong-Off | Capitulation | Risk-Off | 0% |

Trang web hiển thị: trạng thái hôm nay, phân bổ đề xuất, điểm số theo 3 khung thời
gian (dài/trung/ngắn hạn), biểu đồ giá 4 năm tô màu theo trạng thái, backtest so
với buy & hold, và lịch sử các điểm đảo chiều Risk-On/Risk-Off.

## Chạy local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate — Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
# Mở http://127.0.0.1:8000
```

Lần khởi động đầu, server tải giá BTC-USD từ yfinance (từ 2015) ở background;
nếu lỗi hoặc thiếu dữ liệu sẽ tự fallback sang CoinGecko. Request đầu tiên có thể
trả 503 trong ~10–30 giây cho tới khi tính xong — frontend tự thử lại.

### Chạy offline / test

Trỏ biến môi trường `BTC_CSV` tới file CSV có 2 cột `Date,Close`:

```bash
BTC_CSV=/duong/dan/btc.csv uvicorn app.main:app --port 8000
```

### Test

```bash
pytest tests -q
```

Test sinh chuỗi giá random walk trong bộ nhớ, không gọi mạng.

## API

| Endpoint | Nội dung |
|---|---|
| `GET /` | Trang web (index.html) |
| `GET /api/read` | Trạng thái, điểm số, backtest, inflections (không kèm history) |
| `GET /api/history` | Chỉ lịch sử 4 năm (dates, close, state, score, equity) |
| `GET /api/full` | Tất cả |
| `GET /health` | `{ok, cached, error}` |

Khi chưa có dữ liệu, API trả **503** kèm thông báo. CORS mở cho mọi origin trên `/api/*`.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CACHE_TTL_SECONDS` | `3600` | Thời gian cache kết quả mô hình (giây) |
| `BTC_CSV` | _(trống)_ | Đường dẫn CSV `Date,Close` để chạy offline |
| `PORT` | `8000` | Port của server (dùng trong Docker/Render) |

## Chỉnh tham số mô hình

Mọi tham số nằm ở **đầu file `app/model.py`**: trọng số 3 khung (`W_LONG`, `W_MID`,
`W_SHORT`), ngưỡng vào trạng thái (`ENTER_STRONG`, `ENTER_MILD`, `BUFFER`),
bảng phân bổ (`ALLOC`), phí backtest (`FEE`), số năm lịch sử (`HISTORY_YEARS`).

## Deploy

### Render (Blueprint — khuyến nghị)

1. Push repo lên GitHub.
2. Vào Render → **New → Blueprint**, chọn repo. Render đọc `render.yaml`
   (runtime docker, plan free) và tự build.
3. **Lưu ý:** gói free của Render **ngủ sau 15 phút** không có traffic; request
   đánh thức đầu tiên mất ~30–60 giây (cold start + tải dữ liệu giá).

### Railway / Fly.io (Dockerfile)

- **Railway:** New Project → Deploy from GitHub repo; Railway tự nhận `Dockerfile`.
- **Fly.io:** `fly launch` trong thư mục repo (Fly đọc `Dockerfile`), rồi `fly deploy`.

Cả hai chỉ cần set `CACHE_TTL_SECONDS` nếu muốn đổi TTL cache.

## Mô hình (tóm tắt)

- **Dài hạn:** Mayer multiple (giá/SMA200) và độ dốc SMA200 → xu hướng cấu trúc.
- **Trung hạn:** z-score ROC 30 ngày (cửa sổ 365 ngày) + golden cross SMA50/SMA200.
- **Ngắn hạn:** tỷ lệ biến động 7/90 ngày (sốc vol) nhân hướng ROC 7 ngày.
- **Tổng hợp:** `score = 1.2·long + 1.0·mid + 0.8·short`, phân loại 4 trạng thái
  có hysteresis để tránh đảo trạng thái liên tục.
- **Backtest:** tín hiệu ngày *t* áp cho lợi nhuận ngày *t+1* (không nhìn trước),
  phí 0,1% mỗi đơn vị thay đổi phân bổ.

## Hướng phát triển

Ngoài phạm vi hiện tại, có thể cân nhắc: cảnh báo Telegram/email khi đổi trạng thái,
bổ sung dữ liệu on-chain (ví dụ SOPR, MVRV), so sánh nhiều tham số hysteresis,
xuất CSV lịch sử trạng thái, chế độ dark mode.

## Miễn trừ trách nhiệm

Dự án mã nguồn mở tái tạo theo mô tả công khai, không phải sản phẩm của
Glassnode/Swissblock. Kết quả backtest quá khứ không đảm bảo tương lai.
Không phải lời khuyên đầu tư.
