import urllib.request
import zipfile
import io

url = "https://github.com/rajpal-pawar/Crumbs/releases/download/crumbs-v1.0.0/models.zip"
print(f"Downloading from {url}...")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            print("Contents:")
            for info in z.infolist():
                print(f" - {info.filename} ({info.file_size} bytes)")
except Exception as e:
    print(f"Failed: {e}")
