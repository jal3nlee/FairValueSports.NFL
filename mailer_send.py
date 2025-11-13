import os
import requests

MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")

def send_email(to_email, subject, html_content):
    url = "https://api.mailersend.com/v1/email"

    payload = {
        "from": {"email": "youraddress@fairvaluebetting.com"},
        "to": [{"email": to_email}],
        "subject": subject,
        "html": html_content
    }

    headers = {
        "Authorization": f"Bearer {MAILERSEND_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code >= 200 and r.status_code < 300:
        print(f"✔ Sent to {to_email}")
    else:
        print(f"✘ Failed to send to {to_email}: {r.text}")


if __name__ == "__main__":
    # ---- MANUALLY LIST YOUR EMAILS HERE ----
    emails = [
        "friend1@gmail.com",
        "friend2@yahoo.com",
        "friend3@outlook.com",
    ]

    subject = "Fair Value Betting Update"
    html_content = """
    <h1>Hello!</h1>
    <p>This is a manual update from Fair Value Betting.</p>
    """

    print("Sending bulk email...\n")

    for e in emails:
        send_email(e, subject, html_content)

    print("\nDone.")
