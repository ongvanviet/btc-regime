# -*- coding: utf-8 -*-
"""
Cảnh báo khi mô hình đổi trạng thái — Telegram và/hoặc email.

Cách dùng:
1. Trong server: đặt ALERT_ENABLED=1, sau mỗi lần refresh thành công server gọi
   maybe_notify(). Trạng thái đã thông báo lần cuối lưu ở DATA_DIR/last_state.json
   nên mỗi lần đổi chỉ gửi đúng một lần.
2. Chạy độc lập (cron, GitHub Actions):  python -m app.alerts
   (thoát mã 0 nếu chạy xong, 1 nếu tính mô hình lỗi).

Biến môi trường:
  ALERT_ENABLED        1/true để bật trong server (CLI luôn bật)
  TELEGRAM_BOT_TOKEN   token bot Telegram (BotFather)
  TELEGRAM_CHAT_ID     chat id nhận tin (có thể là group, dạng -100...)
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS   máy chủ gửi mail (STARTTLS)
  ALERT_EMAIL_TO       địa chỉ nhận (nhiều địa chỉ cách nhau bằng dấu phẩy)
  ALERT_EMAIL_FROM     địa chỉ gửi (mặc định = SMTP_USER)
  ALERT_ON_STATE       "risk" (mặc định) chỉ báo khi đổi Risk-On/Off,
                       "state" báo mọi lần đổi 1 trong 4 trạng thái
  SITE_URL             link kèm trong tin nhắn
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("riskonbtc.alerts")


def _enabled() -> bool:
    return os.environ.get("ALERT_ENABLED", "").lower() in ("1", "true", "yes")


def _state_file() -> Path:
    from app.data import data_dir
    return data_dir() / "last_state.json"


def read_last_state(path: Optional[Path] = None) -> Optional[dict]:
    path = path or _state_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_last_state(payload: dict, path: Optional[Path] = None) -> None:
    path = path or _state_file()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def should_notify(prev: Optional[dict], cur: dict, mode: str = "risk") -> bool:
    """Quyết định có gửi không. Lần đầu (prev=None) chỉ ghi nhận, không gửi."""
    if prev is None:
        return False
    if mode == "state":
        return prev.get("code") != cur["code"]
    return prev.get("risk") != cur["risk"]


def build_message(result: dict, prev: Optional[dict]) -> tuple[str, str]:
    """Trả (tiêu đề, nội dung) tiếng Việt cho tin nhắn/email."""
    st = result["state"]
    prev_txt = f"{prev['name']} ({prev['risk']})" if prev else "—"
    arrow = "🟢" if st["risk"] == "Risk-On" else "🔴"
    title = f"{arrow} RiskonBTC: {st['name']} — {st['risk']}"
    lines = [
        title,
        f"Ngày chốt: {result['as_of']}  ·  Giá BTC: {result['price']:,.0f} USD",
        f"Trước đó: {prev_txt}",
        f"Phân bổ đề xuất: BTC {int(st['alloc'] * 100)}% / tiền mặt {int((1 - st['alloc']) * 100)}%",
        f"Điểm tổng: {result['score']['total']:+.2f} "
        f"(dài {result['score']['long']:+.2f}, trung {result['score']['mid']:+.2f}, "
        f"ngắn {result['score']['short']:+.2f})",
    ]
    site = os.environ.get("SITE_URL")
    if site:
        lines.append(site)
    lines.append("Không phải lời khuyên đầu tư.")
    return title, "\n".join(lines)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text,
                            "disable_web_page_preview": True}, timeout=20)
    r.raise_for_status()
    return True


def send_email(subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    to = os.environ.get("ALERT_EMAIL_TO")
    if not host or not to:
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    sender = os.environ.get("ALERT_EMAIL_FROM", user)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        if user:
            s.login(user, password)
        s.send_message(msg)
    return True


def maybe_notify(result: dict, force: bool = False, path: Optional[Path] = None) -> dict:
    """So sánh trạng thái hiện tại với lần thông báo trước; gửi nếu đổi.

    Trả về dict {"sent": bool, "channels": [...], "reason": str} để test/CLI dùng.
    """
    if not force and not _enabled():
        return {"sent": False, "channels": [], "reason": "ALERT_ENABLED chưa bật"}

    st = result["state"]
    cur = {"code": st["code"], "name": st["name"], "risk": st["risk"],
           "as_of": result["as_of"], "price": result["price"]}
    prev = read_last_state(path)
    mode = os.environ.get("ALERT_ON_STATE", "risk").lower()

    if not should_notify(prev, cur, mode):
        if prev is None or prev.get("code") != cur["code"]:
            write_last_state(cur, path)     # ghi nhận nhưng không gửi
        return {"sent": False, "channels": [], "reason": "không đổi trạng thái"}

    subject, body = build_message(result, prev)
    channels = []
    errors = []
    for name, fn in (("telegram", lambda: send_telegram(body)),
                     ("email", lambda: send_email(subject, body))):
        try:
            if fn():
                channels.append(name)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning("Gửi %s thất bại: %s", name, e)

    write_last_state(cur, path)
    log.info("Cảnh báo đổi trạng thái: %s → %s, kênh=%s, lỗi=%s",
             prev.get("name") if prev else None, cur["name"], channels, errors)
    return {"sent": bool(channels), "channels": channels,
            "reason": "; ".join(errors) if errors else "đã gửi"}


def main() -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    from app import data, model
    try:
        close, extras, meta = data.load_all()
        result = model.compute(close, extras, meta)
    except Exception as e:
        log.error("Tính mô hình lỗi: %s", e)
        return 1
    out = maybe_notify(result, force=True)
    print(json.dumps({"as_of": result["as_of"], "state": result["state"]["name"],
                      "risk": result["state"]["risk"], **out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
