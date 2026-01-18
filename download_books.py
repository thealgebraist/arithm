import os
import requests

books = [
    1342, 11, 84, 1661, 2701, 145, 1232, 2600, 43, 219, 
    174, 98, 1184, 1513, 2591, 5200, 120, 16, 205, 1952,
    76, 514, 158, 203, 209, 215, 236, 244, 253, 254, 255, 259
]

os.makedirs("books", exist_ok=True)

for book_id in books:
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    filepath = f"books/book_{book_id}.txt"
    if not os.path.exists(filepath):
        print(f"Downloading {book_id}...")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(r.content)
            else:
                # Try another format if UTF-8 not found
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(r.content)
        except Exception as e:
            print(f"Failed to download {book_id}: {e}")

print("Done downloading books.")
