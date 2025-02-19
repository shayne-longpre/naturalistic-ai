import time
import json
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)



def main(input_fpath, output_fpath):
    processed_ids = []

    with open(input_fpath, "r", encoding="utf-8") as outfile:
        for line in outfile:
            data = json.loads(line.strip())
            post_url = data.get("postId")
            processed_ids.append(post_url)


    with open(input_fpath, "r", encoding="utf-8") as infile, open(output_fpath, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            post_url = data.get("postId")

            if post_url in processed_ids:
                continue

            if not post_url:
                print("Skipping entry with missing postId:", data)
                continue

            driver.get(post_url)
            time.sleep(2)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            text_content = soup.get_text(separator="\n", strip=True)

            data["parsed_html"] = text_content
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIArchive crawling code.")
    parser.add_argument(
        "--input_fpath", 
        type=str, 
        required=True, 
        help="Path to the input json file."
    )
    parser.add_argument(
        "--output_fpath", 
        type=str, 
        required=True, 
        help="Save path to the output jsonl file."
    )
    
    args = parser.parse_args()
    main(args.input_fpath, args.output_fpath)
