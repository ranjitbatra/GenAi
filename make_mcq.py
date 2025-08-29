from pathlib import Path
import csv

Path("data/eval").mkdir(parents=True, exist_ok=True)

rows = [
    ["What is the standard U.S. shipping time after processing?",
     "1–2 business days","3–5 business days","7–10 business days","Same-day",
     "B","From 'Shipping Times': Standard shipping takes 3–5 business days.","Shipping Times","shipping-times"],
    ["Within how many days can most items be returned for a refund?",
     "7 days","14 days","30 days","60 days",
     "C","From 'Returns & Refund Policy': return most items within 30 days.","Returns & Refund Policy","policy-returns"],
    ["Who is responsible for customs duties on international orders?",
     "The sender","The carrier","The recipient","The marketplace",
     "C","From 'International Shipping & Customs': duties are the recipient’s responsibility.","International Shipping & Customs","intl-customs"],
    ["What does the one-year warranty cover?",
     "Accidental damage","Defects in materials and workmanship","Loss or theft","Cosmetic wear",
     "B","From 'Warranty Coverage': it covers defects in materials and workmanship.","Warranty Coverage","warranty"],
    ["If you can’t log in, which first step is recommended?",
     "Create a new account","Contact your bank","Reset your password","Disable 2FA",
     "C","From 'Account Login Troubleshooting': reset your password.","Account Login Troubleshooting","account-login"],
]

out = Path("data/eval/mcqs.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question","A","B","C","D","correct","why","title","id"])
    w.writerows(rows)

print("Wrote", out.resolve())
