# RiskonBTC (btc-regime)

Web app mã nguồn mở đọc **trạng thái rủi ro (risk regime) của Bitcoin** mỗi ngày,
lấy cảm hứng từ các mô tả công khai về sản phẩm cùng loại. Đây là bản tái tạo theo
mô tả, **không** dùng thuật toán gốc và **không** liên quan tới bất kỳ sản phẩm thương mại nào.

Mỗi ngày (sau khi nến ngày UTC đóng) mô hình xếp Bitcoin vào 1 trong 4 trạng thái:

| Mã | Trạng thái | Nhãn | Risk | Phân bổ BTC |
|---:|---|---|---|---:|
| 2 | Strong-On | Expansion | Risk-On | 100% |
| 1 | Mild-On | Accumulation | Risk-On | 60% |
| −1 | Mild-Off | Caution | Risk-Off | 25% |
| −2 | Strong-Off | Capitulation | Risk-Off | 0% |

Trang web hiển thị: trạng thái đã chốt và **xem trước intraday** (nến hôm nay chưa đóng),
phân bổ đề xuất, điểm số **5 thành phần** (dài / trung / ngắn hạn, on-chain MVRV, vĩ mô USD),
biểu đồ giá tô màu theo trạng thái (1/2/4 năm), backtest so với buy & hold kèm drawdown,
lợi nhuận theo năm, thống kê hành vi trạng thái, công cụ **so sánh tham số**, lịch sử điểm
đảo chiều, xuất CSV, giao diện sáng/tối.

## Chạy local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate — Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
# Mở http://127.0.0.1:8000
```

Lần khởi động đầu, server đọc file hạt giống `app/seed/btc_seed.csv` (giá từ 2015) rồi
chỉ tải **phần còn thiếu** (vài ngày gần nhất) từ Binance → Coinbase → Coin Metrics → yfinance.
Giá được lưu vào `DATA_DIR/prices.csv` để các lần sau chạy nhanh hơn. Nếu mọi nguồn đều lỗi,
app vẫn chạy với dữ liệu đã lưu và hiển thị cảnh báo "dữ liệu cũ".

### Chạy offline / test

Trỏ biến môi trường `BTC_CSV` tới file CSV có 2 cột `Date,Close`:

```bash
BTC_CSV=/duong/dan/btc.csv DISABLE_EXTRAS=1 uvicorn app.main:app --port 8000
```

### Test

```bash
pytest tests -q
```

Test dùng chuỗi giá random walk trong bộ nhớ và mock lớp dữ liệu, không gọi mạng.

## API

| Endpoint | Nội dung |
|---|---|
| `GET /` | Trang web (index.html) |
| `GET /api/read` | Trạng thái, điểm số, backtest, intraday, meta (không kèm history) |
| `GET /api/history` | Lịch sử 4 năm (dates, close, state, score, equity, drawdown) |
| `GET /api/full` | Tất cả |
| `GET /api/history.csv` | Lịch sử 4 năm dạng CSV để tải về |
| `GET /api/backtest?smooth_days=7&confirm_days=2&buffer=0.4&...` | Chạy lại backtest với tham số khác (xem `PARAM_BOUNDS` trong `app/model.py`) |
| `GET /health` | `{ok, cached, age_seconds, refreshing, error}` — trả 503 khi dữ liệu cũ quá 24 giờ hoặc chưa từng tính được |

Khi chưa có dữ liệu, API trả **503** kèm `Retry-After`. Phản hồi `/api/*` được nén gzip và có
`Cache-Control: public, max-age=300`. CORS mở cho mọi origin.

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CACHE_TTL_SECONDS` | `3600` | Thời gian cache kết quả mô hình (giây) |
| `STALE_AFTER_SECONDS` | `86400` | Sau bao lâu không refresh được thì `/health` báo lỗi |
| `API_MAX_AGE` | `300` | `max-age` trong Cache-Control của `/api/*` |
| `DATA_DIR` | `./data` | Thư mục lưu giá, MVRV, DXY và trạng thái cảnh báo |
| `BTC_CSV` | _(trống)_ | Đường dẫn CSV `Date,Close` để chạy offline |
| `DISABLE_EXTRAS` | _(trống)_ | `1` để không tải MVRV/DXY (mô hình tự bỏ 2 thành phần này) |
| `LOG_LEVEL` | `INFO` | Mức log |
| `PORT` | `8000` | Port của server (Docker/Render) |

### Cảnh báo đổi trạng thái

| Biến | Ý nghĩa |
|---|---|
| `ALERT_ENABLED` | `1` để server gửi cảnh báo sau mỗi lần refresh (CLI luôn bật) |
| `ALERT_ON_STATE` | `risk` (mặc định): chỉ báo khi đổi Risk-On/Off; `state`: báo mọi lần đổi 1 trong 4 trạng thái |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Gửi qua Telegram |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM` | Gửi qua email (STARTTLS) |
| `SITE_URL` | Link kèm trong tin nhắn |

Trạng thái đã thông báo lần cuối lưu ở `DATA_DIR/last_state.json`, nên mỗi lần đổi chỉ gửi đúng một lần.

Chạy độc lập (cron):

```bash
python -m app.alerts
```

Repo có sẵn workflow `.github/workflows/daily-alert.yml` chạy lúc 00:20 UTC mỗi ngày; chỉ cần
thêm secrets `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (và/hoặc SMTP) vào GitHub repo.

## Mô hình

Tham số nằm trong dataclass `Params` ở đầu `app/model.py` (mặc định = `DEFAULT_PARAMS`).

- **Dài hạn (w=1.2):** Mayer multiple (giá/SMA200) và độ dốc SMA200 → xu hướng cấu trúc.
- **Trung hạn (w=1.0):** z-score ROC 30 ngày (cửa sổ 365 ngày) + golden cross SMA50/SMA200.
- **Ngắn hạn (w=0.8):** `clip(ROC7 / 10%, −1, 1)` nhân sốc biến động (vol 7 ngày / 90 ngày).
  Dùng giá trị liên tục thay vì dấu của ROC7 để tránh nhảy bậc.
- **On-chain (w=0.6):** z-score 365 ngày của MVRV (Coin Metrics community API), trừ 0.5 khi MVRV > 3.
- **Vĩ mô (w=0.4):** −z-score ROC 20 ngày của chỉ số USD trade-weighted (FRED `DTWEXBGS`).
- **Tổng hợp:** trung bình có trọng số của các thành phần **có dữ liệu**, chuẩn hoá về thang ±3,
  rồi làm mượt EMA 7 ngày. Thiếu MVRV/DXY thì tự bỏ thành phần đó, thang điểm không đổi.
- **Phân loại:** 4 trạng thái có hysteresis (ngưỡng Strong 1.8, Mild 0.3, đệm 0.4) và
  **xác nhận 2 ngày liên tiếp** mới đổi trạng thái.
- **Nến chưa đóng:** trạng thái chính thức chỉ dùng nến đã đóng (ngày < hôm nay UTC); nến hôm nay
  chỉ dùng để "xem trước intraday".
- **Backtest:** tín hiệu ngày *t* áp cho lợi nhuận ngày *t+1* (không nhìn trước), phí 0,1% mỗi đơn
  vị thay đổi phân bổ.

### Vì sao làm mượt + xác nhận

Trên dữ liệu thật (2016–2026), phiên bản cũ (dấu ROC7, không làm mượt) đổi trạng thái ~40 lần/năm,
regime trung vị 4 ngày. Với EMA 7 ngày + xác nhận 2 ngày: ~9 lần/năm, regime trung vị ~32 ngày,
max drawdown giảm từ −62% xuống −52%, CAGR không giảm. Thêm on-chain + vĩ mô giảm số lần đảo
Risk-On/Off xuống ~1,7 lần/năm nhưng CAGR thấp hơn vài điểm phần trăm. Có thể thử lại các bộ tham
số khác ngay trên trang web (mục "So sánh tham số") hoặc qua `/api/backtest`.

## Deploy

### Render (Blueprint)

1. Push repo lên GitHub.
2. Vào Render → **New → Blueprint**, chọn repo. Render đọc `render.yaml` (runtime docker, plan free).
3. Gói free **ngủ sau 15 phút** không có traffic; nhờ file hạt giống, request đánh thức chỉ mất
   vài giây để tải phần giá còn thiếu. Lưu ý Binance chặn IP Mỹ; app tự chuyển sang Coinbase/Coin Metrics.

### Fly.io / Railway (Dockerfile)

- **Fly.io:** `fly launch` rồi `fly deploy`. Muốn giữ lịch sử giá qua các lần deploy, tạo volume và bỏ
  comment mục `[mounts]` trong `fly.toml`. Bật cảnh báo: `fly secrets set ALERT_ENABLED=1 TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...`.
- **Railway:** New Project → Deploy from GitHub repo; Railway tự nhận `Dockerfile`.

Docker image có `HEALTHCHECK` gọi `/health`; production chỉ cài `requirements.txt`
(pytest/httpx nằm trong `requirements-dev.txt`).

## Miễn trừ trách nhiệm

Dự án mã nguồn mở tái tạo theo mô tả công khai. Kết quả backtest quá khứ không đảm bảo tương lai.
Không phải lời khuyên đầu tư.
