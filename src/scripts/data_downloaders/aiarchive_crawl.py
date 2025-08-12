import time
import json
import argparse
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--start-maximized")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


def get_processed_ids(output_fpath):
    """Read the output file and extract all already processed postIds."""
    processed_ids = set()
    
    if not os.path.exists(output_fpath):
        print(f"Output file {output_fpath} does not exist. Starting fresh.")
        return processed_ids
    
    print(f"Reading existing output file: {output_fpath}")
    try:
        with open(output_fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        post_id = data.get("postId")
                        if post_id:
                            processed_ids.add(post_id)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line: {e}")
                        continue
    except Exception as e:
        print(f"Error reading existing output file: {e}")
    
    print(f"Found {len(processed_ids)} already processed IDs")
    return processed_ids


def main(input_fpath, output_fpath):
    # Get already processed IDs
    processed_ids = get_processed_ids(output_fpath)
    
    # Open output file in append mode
    open_mode = "a" if processed_ids else "w"
    
    with open(input_fpath, "r", encoding="utf-8") as infile, open(output_fpath, open_mode, encoding="utf-8") as outfile:
        total_processed = 0
        total_skipped = 0
        
        for idx, line in enumerate(infile):
            if idx % 100 == 0:
                print(f"Index: {idx} (Processed: {total_processed}, Skipped: {total_skipped})")
            
            data = json.loads(line.strip())
            post_url = data.get("postId")
            created = data.get("created")

            if not post_url:
                print("Skipping entry with missing postId:", data)
                continue
            
            # Skip if already processed
            if post_url in processed_ids:
                total_skipped += 1
                if total_skipped % 100 == 0:
                    print(f"Skipped {total_skipped} already processed entries...")
                continue
            
            # print(f"Processing new entry {idx}: {post_url}")
            total_processed += 1

            conversation_data = {"postId": post_url, "created": created}
            index = 1
            max_empty_attempts = 5
            empty_attempts = 0

            try:
                driver.get(post_url)
                time.sleep(2)

                page_source = driver.page_source
                if 'font-claude-message' in page_source:
                    model_name = "Claude"
                elif 'bg-[#10A37F]' in page_source:
                    model_name = "ChatGPT"
                elif '#FFC107' in page_source:
                    model_name = "Bard"
                else:
                    model_name = "Unknown"

                conversation_data["model"] = model_name

                while empty_attempts < max_empty_attempts:
                    try:
                        # Extract Prompt (even index turns)
                        prompt_xpath = f'//*[@id="root"]/div[1]/div[2]/div[1]/div[3]/div[2]/div[{index}]/div/div/div/p'
                        try:
                            prompt_element = driver.find_element(By.XPATH, prompt_xpath)
                            prompt_text = prompt_element.text.strip()
                            conversation_data[f"Turn {index - 1}"] = prompt_text
                        except NoSuchElementException:
                            empty_attempts += 1
                            index += 2
                            continue

                        # Extract Response (odd index turns)
                        response_xpath = f'//*[@id="root"]/div[1]/div[2]/div[1]/div[3]/div[2]/div[{index+1}]/div/div/div[2]/div/div/div'
                        try:
                            response_element = driver.find_element(By.XPATH, response_xpath)
                            response_text = response_element.text.strip()
                            conversation_data[f"Turn {index}"] = response_text
                        except:
                            try:
                                response_xpath = f'//*[@id="root"]/div[1]/div[2]/div[1]/div[3]/div[2]/div[{index+1}]/div/div/div[2]/div/div'
                                response_element = driver.find_element(By.XPATH, response_xpath)
                                response_text = response_element.text.strip()
                                conversation_data[f"Turn {index}"] = response_text
                            except:
                                try:
                                    response_xpath = f'//*[@id="root"]/div[1]/div[2]/div[1]/div[3]/div[2]/div[{index+1}]/div/div/div[2]/div/div'
                                    response_element = driver.find_element(By.XPATH, response_xpath)
                                    response_text = response_element.text.strip()
                                    conversation_data[f"Turn {index}"] = response_text
                                except:
                                    conversation_data[f"Turn {index}"] = ""

                        index += 2  # Move to the next pair
                        empty_attempts = 0

                    except Exception as e:
                        print(f"Error extracting conversation: {e}")
                        break

                outfile.write(json.dumps(conversation_data, ensure_ascii=False) + "\n")
                outfile.flush()  # Ensure data is written immediately
                
            except Exception as e:
                print(f"Error processing {post_url}: {e}")
                # Optionally, you might want to write a partial record or error log
                conversation_data["error"] = str(e)
                outfile.write(json.dumps(conversation_data, ensure_ascii=False) + "\n")
                outfile.flush()
        
        print(f"\nProcessing complete!")
        print(f"Total new entries processed: {total_processed}")
        print(f"Total entries skipped (already processed): {total_skipped}")
    
    driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIArchive crawling code with resume capability.")
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