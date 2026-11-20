"""Email the PDF report over SMTP. All settings come from environment variables, never from code or git.

    EQR_EMAIL_TO         recipient (required; if unset, email is skipped)
    EQR_SMTP_USER        SMTP login (required)
    EQR_SMTP_PASSWORD    SMTP password / app password (required)
    EQR_SMTP_HOST        default smtp.gmail.com
    EQR_SMTP_PORT        default 587 (STARTTLS); 465 uses implicit TLS
    EQR_EMAIL_FROM       default: EQR_SMTP_USER
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


class NotConfigured(Exception):
    pass


def send_report(pdf_path: Path, subject: str, body: str, env=os.environ, smtp_factory=None) -> str:
    to, user, pw = env.get("EQR_EMAIL_TO"), env.get("EQR_SMTP_USER"), env.get("EQR_SMTP_PASSWORD")
    missing = [n for n, v in (("EQR_EMAIL_TO", to), ("EQR_SMTP_USER", user), ("EQR_SMTP_PASSWORD", pw)) if not v]
    if missing:
        raise NotConfigured(f"set {', '.join(missing)} to enable emailing (see README)")
    host, port = env.get("EQR_SMTP_HOST", "smtp.gmail.com"), int(env.get("EQR_SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = env.get("EQR_EMAIL_FROM", user), to, subject
    msg.set_content(body)
    msg.add_attachment(Path(pdf_path).read_bytes(), maintype="application", subtype="pdf", filename=Path(pdf_path).name)

    ctx = ssl.create_default_context()
    if smtp_factory:  # tests inject a fake
        server = smtp_factory(host, port)
    elif port == 465:
        server = smtplib.SMTP_SSL(host, port, context=ctx, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
    with server as s:
        if port != 465 and not smtp_factory:
            s.starttls(context=ctx)
        s.login(user, pw)
        s.send_message(msg)
    return to
