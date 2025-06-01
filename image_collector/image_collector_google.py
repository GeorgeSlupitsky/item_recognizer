from icrawler.builtin import GoogleImageCrawler

def download_images(keyword, max_num=200, output_dir='dataset2'):
    crawler = GoogleImageCrawler(storage={'root_dir': f'{output_dir}/{keyword}'})
    crawler.crawl(keyword=keyword, max_num=max_num)

if __name__ == '__main__':
    categories = [
        "vinyls",
        "coins",
        "stamps",
        "banknotes",
        "drumsticks",
        "trading cards",
        "comic books",
        "CD covers",
        "books covers"
    ]
    
    for category in categories:
        print(f"Downloading: {category}")
        download_images(category, max_num=200)
