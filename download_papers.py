import arxiv
import os

os.makedirs("data", exist_ok=True)

search = arxiv.Search(
    query="AI healthcare OR medical imaging OR clinical AI",
    max_results=100
)

count = 0

for result in search.results():
    try:
        result.download_pdf(dirpath="data")
        count += 1
        print(f"Downloaded {count}")
    except Exception as e:
        print("Failed:", e)

print("TOTAL DOWNLOADED:", count)