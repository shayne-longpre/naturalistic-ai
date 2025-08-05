## AIArchive Crawler
This is the code for the crawler for <a href="https://aiarchives.org/">AIArchive website</a>. 

### Prerequisties
Using Python >= 3.8:
```bash
pip install -r requirements.txt
```

You will need input jsonl file in the following format:
```json
{"postId": "https://aiarchives.org/id/XueuO7I8s0AJ3IhngtFy", "created": 20241103225454}
{"postId": "https://aiarchives.org/id/3DCVoToHd33Ckt7R2mzY", "created": 20241103225320}
...
```

### How to run?
You can run the code by using the following command:
```bash
python aiarchive_crawl.py --input_fpath aiarchive.jsonl --output_fpath aiarchive_parsed.jsonl
```

This code will visit each postId in the input file and save the parsed text content to the output json file.
```json
{
    "postId": "https://aiarchives.org/id/AXyMRUssrmrHIIj4fJFc",
    "created": 20240124171207,
    "model": "ChatGPT",
    "Turn 0": "You are an experienced dog trainer. Please provide some best tips to someone who has never trained dogs to begin training their own dog. Provide step-by-step instructions on how to begin training their dog at home. Some of the instructions should include: potty training and basic commands.”",
    "Turn 1": "Certainly! Training your dog at home can be a rewarding experience. Here are some step-by-step instructions on how to begin training your dog, including potty training and basic commands:\n1. Start with Basic Commands:\nBegin with basic commands like \"sit,\" \"stay,\" \"come,\" and \"heel.\" These commands are essential for your dog's safety and behavior.\n2. Use Positive Reinforcement:\nPositive reinforcement is crucial in dog training. Reward your dog with treats, praise, and affection when they perform the desired behavior. This encourages them to repeat it.\n3. Consistency is Key:\nBe consistent with your commands and rewards. Use the same words and gestures each time, so your dog understands what you expect.\n4. Set a Routine:\nDogs thrive on routine. Establish a daily schedule for feeding, potty breaks, and training sessions. This helps them predict what's expected and when.\n5. Potty Training:\nFor potty training, take your dog outside:\nAfter waking up in the"
}
```