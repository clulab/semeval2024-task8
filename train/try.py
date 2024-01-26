import requests
notif = (
    "herllo world"
)
requests.post(
    "https://ntfy.sh/mhrnlpmodels",
    data=notif.encode(encoding="utf-8"),
    headers={"Priority": "5"},
)