import os
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def setup_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--log-level=3')
    return webdriver.Chrome(options=options)

def get_ebay_image_urls(keyword, max_images=200):
    driver = setup_driver()
    query = keyword.replace(" ", "+")
    url = f"https://www.ebay.com/sch/i.html?_nkw={query}"
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    images = soup.find_all('img', {"src": True})
    image_urls = []

    for img in images:
        src = img['src']
        if "thumbs" in src or "1x1.gif" in src or not src.startswith("http"):
            continue
        if len(image_urls) >= max_images:
            break
        image_urls.append(src)

    return image_urls

def download_images(image_urls, folder):
    os.makedirs(folder, exist_ok=True)
    for i, url in enumerate(tqdm(image_urls, desc=f"Downloading to {folder}")):
        try:
            response = requests.get(url, timeout=5)
            with open(os.path.join(folder, f"{i+1:03d}.jpg"), 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading {url}: {e}")

if __name__ == '__main__':
    categories = {
        "cds": "CD"
    }

    for folder, query in categories.items():
        print(f"\nScraping eBay for: {query}")
        urls = get_ebay_image_urls(query, max_images=200)
        download_images(urls, folder=os.path.join('dataset3', folder))
