import os
import requests

MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")

def send_template_email(to_email, template_id, variables=None):
    url = "https://api.mailersend.com/v1/email"

    payload = {
        "from": {"email": "youraddress@fairvaluebetting.com"},
        "to": [{"email": to_email}],
        "template_id": template_id,
        "variables": variables or []
    }

    headers = {
        "Authorization": f"Bearer {MAILERSEND_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(url, json=payload, headers=headers)

    if 200 <= r.status_code < 300:
        print(f"✔ Template sent to {to_email}")
    else:
        print(f"✘ Failed for {to_email}: {r.status_code} | {r.text}")


if __name__ == "__main__":
    # ----------------------------------------
    # 🔥 ENTER YOUR EMAILS HERE
    # ----------------------------------------
    emails = [
        "jalenlee04@gmail.com",
    ]
    # ----------------------------------------

    # 🔥 Your MailerSend template ID
    TEMPLATE_ID = "z3m5jgryx604dpyo"

    # Optional dynamic variables used inside MailerSend templates
    # These match {{name}} and {{custom_message}} in your template
    template_variables = [
        {
            "email": e,
            "substitutions": [
                {"var": "name", "value": e.split("@")[0].title()},
                {"var": "custom_message", "value": "Welcome to Fair Value Betting!"}
            ]
        }
        for e in emails
    ]

    print("Sending template emails...\n")

    for entry in template_variables:
        send_template_email(
            entry["email"],
            TEMPLATE_ID,
            variables=[{
                "email": entry["email"],
                "substitutions": entry["substitutions"]
            }]
        )

    print("\nDone.")
