import os
import csv
import re

# Input and output folders
folder = "MRDS"
output_folder = "MRDS"
os.makedirs(output_folder, exist_ok=True)

# Pattern that detects quoted text
QUOTE_PATTERN = r'"[^"]*"'

for filename in os.listdir(folder):
    if filename.lower().endswith(".txt"):
        txt_path = os.path.join(folder, filename)
        csv_path = os.path.join(output_folder, filename.replace(".txt", ".csv"))

        print(f"Converting: {filename}")

        rows = []
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                # Protect quoted text before splitting
                placeholders = {}
                for i, match in enumerate(re.findall(QUOTE_PATTERN, line)):
                    key = f"@@Q{i}@@"
                    placeholders[key] = match
                    line = line.replace(match, key)

                # Normalize separators
                line = line.replace("\t", "|")
                line = re.sub(r'\s{2,}', '|', line)

                # Split fields and preserve blanks
                parts = line.split("|")

                # Restore quotes and clean values
                cleaned = []
                for p in parts:
                    p = p.strip()
                    if p in placeholders:
                        p = placeholders[p]
                    p = p.replace('"', "").strip()
                    cleaned.append(p)

                rows.append(cleaned)

        # Pad rows to equal width
        max_len = max(len(r) for r in rows)
        for r in rows:
            r.extend([""] * (max_len - len(r)))

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerows(rows)

print("Done converting files.")