# mailer_send.py
import os
import requests

MAILERSEND_KEY = os.getenv("MAILERSEND_API_KEY")

def send_email(to_email: str, subject: str, html_content: str):
    """
    Send an email using the MailerSend API.

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        html_content: HTML body content of the email.

    Returns:
        Response object from MailerSend.
    """
    if not MAILERSEND_KEY:
        raise ValueError("MAILERSEND_API_KEY is missing from environment variables.")

    url = "https://api.mailersend.com/v1/email"

    payload = {
        "from": {
            "email": "support@fairvaluebetting.com",
            "name": "Fair Value Betting"
        },
        "to": [{"email": to_email}],
        "subject": subject,
        "html": html_content
    }

    headers = {
        "Authorization": f"Bearer {MAILERSEND_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response
