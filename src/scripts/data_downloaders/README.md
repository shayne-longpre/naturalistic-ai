## AIArchive Crawler
This is the code for the crawler for <a href="https://aiarchives.org/">AIArchive website</a>. 

### Prerequisties
You will need input jsonl file where in the following format:
```json
{"postId": "https://aiarchives.org/id/XueuO7I8s0AJ3IhngtFy", "created": 20241103225454}
{"postId": "https://aiarchives.org/id/3DCVoToHd33Ckt7R2mzY", "created": 20241103225320}
...
```

This code will visit each postId in the input file and save the parsed text content to the output jsonl file.
```json
{"postId": "https://aiarchives.org/id/0NzReASZTM9z8dxjrGh3", "created": 20241103101421, "parsed_html": "Identify 5 data breaches since 2016 that involved advanced cyberattack methods, such as zero-day vulnerabilities, supply chain attacks, or insider threats. [...]"}
```

### How to run?
You can run the code by the following command:
```bash
python aiarchive_crawl.py --input_fpath aiarchive.jsonl --output_fpath aiarchive_parsed.jsonl
```