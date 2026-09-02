import os
import re
import subprocess
import time
from urllib.parse import quote

TAB_FILE = "DARTIS_2019.tab"

BASE_URL = "https://download.pangaea.de/dataset/980773/files"

IMAGE_DIR = "images"
XML_DIR = "annotations"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)


# ---------------------------------------------------------
# Read filenames
# ---------------------------------------------------------

print("Reading DARTIS_2019.tab...")

files = set()

with open(TAB_FILE, "r", encoding="utf-8") as f:

    for line in f:

        matches = re.findall(
            r'(?i)([^\t\s"]+\.(?:jpg|jpeg|xml))',
            line
        )

        for filename in matches:
            files.add(filename)


jpg_files = sorted(
    f for f in files
    if f.lower().endswith((".jpg", ".jpeg"))
)

xml_files = sorted(
    f for f in files
    if f.lower().endswith(".xml")
)


print()
print("======================================")
print("DARTIS 2019")
print("======================================")
print(f"JPG files : {len(jpg_files)}")
print(f"XML files : {len(xml_files)}")
print("======================================")
print()


# ---------------------------------------------------------
# Download one file with strong retry logic
# ---------------------------------------------------------

def download_file(filename, output_dir):

    output_path = os.path.join(output_dir, filename)

    # Already downloaded
    if os.path.exists(output_path):

        print(f"[SKIP] {filename}")

        return True


    encoded_filename = quote(filename)

    url = f"{BASE_URL}/{encoded_filename}"


    # Try up to 10 times
    for attempt in range(1, 11):

        print(
            f"[TRY {attempt}/10] {filename}"
        )


        command = [
            "curl",

            "-L",

            "--fail",

            "--silent",

            "--show-error",

            "--retry", "0",

            "--connect-timeout", "30",

            "--max-time", "180",

            "-o", output_path,

            url
        ]


        result = subprocess.run(command)


        if result.returncode == 0:

            print(f"[OK] {filename}")

            return True


        # Remove incomplete file
        if os.path.exists(output_path):

            try:
                os.remove(output_path)

            except OSError:
                pass


        # Wait longer after each failure
        wait_time = min(60, attempt * 5)

        print(
            f"Connection failed. "
            f"Waiting {wait_time} seconds..."
        )

        time.sleep(wait_time)


    print(f"[FAILED AFTER 10 ATTEMPTS] {filename}")

    return False


# ---------------------------------------------------------
# JPG
# ---------------------------------------------------------

failed_jpg = []

print()
print("Downloading JPG images...")
print("--------------------------------------")


for i, filename in enumerate(jpg_files, start=1):

    print()
    print(f"JPG [{i}/{len(jpg_files)}]")

    success = download_file(
        filename,
        IMAGE_DIR
    )

    if not success:

        failed_jpg.append(filename)

    # Small delay between downloads
    time.sleep(2)


# ---------------------------------------------------------
# XML
# ---------------------------------------------------------

failed_xml = []

print()
print("Downloading XML annotations...")
print("--------------------------------------")


for i, filename in enumerate(xml_files, start=1):

    print()
    print(f"XML [{i}/{len(xml_files)}]")

    success = download_file(
        filename,
        XML_DIR
    )

    if not success:

        failed_xml.append(filename)

    time.sleep(2)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("======================================")
print("DOWNLOAD FINISHED")
print("======================================")

print(f"JPG total   : {len(jpg_files)}")
print(f"JPG failed  : {len(failed_jpg)}")

print(f"XML total   : {len(xml_files)}")
print(f"XML failed  : {len(failed_xml)}")

print("======================================")


# ---------------------------------------------------------
# Save failures
# ---------------------------------------------------------

if failed_jpg or failed_xml:

    with open(
        "failed_downloads.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for filename in failed_jpg:
            f.write(filename + "\n")

        for filename in failed_xml:
            f.write(filename + "\n")

    print()
    print("Failed files saved in:")
    print("failed_downloads.txt")
