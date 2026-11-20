"""Email the PDF report. Two methods, chosen with EQR_EMAIL_METHOD. All settings come from environment variables,
never from code or git.

    EQR_EMAIL_TO         recipient (required for both methods; if unset, email is skipped)
    EQR_EMAIL_METHOD     "smtp" (default) or "macos-mail"

  smtp:
    EQR_SMTP_USER        SMTP login (required)
    EQR_SMTP_PASSWORD    SMTP password / app password (required)
    EQR_SMTP_HOST        default smtp.gmail.com
    EQR_SMTP_PORT        default 587 (STARTTLS); 465 uses implicit TLS
    EQR_EMAIL_FROM       default: EQR_SMTP_USER

  macos-mail: sends through the Mail app already signed in on this Mac, so no password is stored anywhere.
    EQR_EMAIL_FROM       optional: which Mail account address to send from
    The first run makes macOS ask whether Terminal may control Mail; allow it.
"""
from __future__ import annotations

import os
import smtplib
import ssl
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path


class NotConfigured(Exception):
    pass


# Arguments arrive via argv (no string interpolation into the script), so subjects and bodies are injection-safe.
MAIL_SCRIPT = """on run argv
    set theSubject to item 1 of argv
    set theBody to item 2 of argv
    set theTo to item 3 of argv
    set thePath to item 4 of argv
    set theFrom to item 5 of argv
    tell application "Mail"
        set m to make new outgoing message with properties {subject:theSubject, content:theBody & return & return, visible:false}
        if theFrom is not "" then set sender of m to theFrom
        tell m
            make new to recipient at end of to recipients with properties {address:theTo}
            tell content to make new attachment with properties {file name:(POSIX file thePath)} at after the last paragraph
        end tell
        send m
    end tell
end run
"""


def _send_macos_mail(pdf_path: Path, subject: str, body: str, env, run, platform) -> str:
    if platform != "darwin":
        raise NotConfigured("EQR_EMAIL_METHOD=macos-mail only works on macOS")
    to = env.get("EQR_EMAIL_TO")
    if not to:
        raise NotConfigured("set EQR_EMAIL_TO to enable emailing (see README)")
    proc = run(["osascript", "-", subject, body, to, str(Path(pdf_path).resolve()), env.get("EQR_EMAIL_FROM", "")],
               input=MAIL_SCRIPT, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"Mail via AppleScript failed: {proc.stderr.strip()[:300]}")
    return to


def send_report(pdf_path: Path, subject: str, body: str, env=os.environ, smtp_factory=None,
                run=subprocess.run, platform: str = sys.platform) -> str:
    if env.get("EQR_EMAIL_METHOD", "smtp").lower() in ("macos-mail", "mail", "applescript"):
        return _send_macos_mail(pdf_path, subject, body, env, run, platform)

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
