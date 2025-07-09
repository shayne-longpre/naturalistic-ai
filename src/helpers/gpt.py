import os
import openai
import json
import asyncio
import aiohttp
import tiktoken
from dotenv import load_dotenv


class GPT(object):
    """
    This class provides an interface to the OpenAI API to perform text generation tasks using various GPT models. 
    It allows for synchronous and asynchronous interactions with the API to generate responses for a given prompt 
    using specified model parameters. The class is designed to handle multiple prompt configurations and includes 
    methods to load environment variables, handle API authentication, and process batches of prompts for efficiency.
    
    Attributes:
        language_code (str): Language of the prompts to be used, default is English ('en').
        model (str): Identifier for the OpenAI GPT model to be used.
        prompt_id (str): Identifier for the specific prompt configuration loaded from JSON.
        SYSTEM_PROMPT (str): Default system prompt defining the role of the assistant.
        USER_PROMPT_1 (str): First user prompt to initiate the conversation.
        ASSISTANT_PROMPT_1 (str): Assistant's initial response in the conversation flow.
        GUIDELINES_PROMPT_TEMPLATE (str): Loaded guidelines prompt for guiding the assistant's responses.
    """
    def __init__(self, 
                 base_url='https://api.openai.com/v1/chat/completions',
                 key_env='OPENAI_API_KEY',
                 language='en',
                 model='gpt-3.5-turbo',
                 prompt='',
                 temperature=1,
                 top_p=1,
                 cache_id='v0',
                 disable_token_counting=False,):

        self.load_API_key(key_env, base_url)
        self.api_key = os.environ.get(key_env)
        self.base_url = base_url
        self.language_code = language
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.disable_token_counting = disable_token_counting
        if self.disable_token_counting:
            print("Token counting is disabled.")
        else:
            print("Token counting is enabled.")

        self.SYSTEM_PROMPT = prompt


        # self.USER_PROMPT_1 = "Are you clear about your role?"
        # if 'system-prompt' in self.prompt_id:
        #     self.SYSTEM_PROMPT = self.load_prompt_from_json()
        #     self.ASSISTANT_PROMPT_1 = "Yes, and I understand to return only a dictionary with my verdict and exact text from the TOS document as evidence. I will not add any other explanation. Please go ahead and provide me with the TOS document." 
        #     self.GUIDELINES_PROMPT_TEMPLATE = "Here is the TOS document: {}"
        # else:
        #     self.SYSTEM_PROMPT = "You are a smart and intelligent legal assistant. I will provide you with the Terms of Use/Service document for a website and you will answer legal questions about that document."
        #     self.ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your task. Please provide me with the necessary information to get started."
        #     self.GUIDELINES_PROMPT_TEMPLATE = self.load_prompt_from_json()

        self.cache_file_path = f'data/cache/cache_{self.model}_{self.temperature}_{self.top_p}_{cache_id}.json'
        self.load_cache()

        self.token_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0
        }

    def load_API_key(self, var='OPENAI_API_KEY', base_url='https://api.openai.com/v1/chat/completions'):
        try:
            load_dotenv(dotenv_path='data/.env')
            openai.api_key = os.environ[var]
            openai.api_base = base_url
            print("API key and base_url successfully loaded!")
        except KeyError:
            print(f"Error: {var} environment variable is not set.")
        except openai.error.AuthenticationError:
            print("Error: Incorrect API key.")

    # def load_prompt_from_json(self):
    #     """Loads a specific prompt from a JSON file based on a provided key, defaults to 'scraping-policy' prompt.
    #         Other keys supported are: "scraping-policy", "AI-policy", "competing-services", "illicit-content", "type-of-license"
    #     """
    #     try:
    #         with open('data/prompt_templates.json', 'r') as file:
    #             prompts = json.load(file)["prompts"]
    #             for prompt in prompts:
    #                 if prompt["id"] == self.prompt_id:
    #                     return prompt["content"]
    #     except FileNotFoundError:
    #         print("prompt_templates.json file not found.")
    #     except json.JSONDecodeError:
    #         print("Error decoding prompts.json.")

        # return default prompt (scraping-policy) if no specific prompt is found or in case of error
        # print(f"Prompt with key '{self.prompt_id}' not found. Using default prompt: 'scraping-policy'.")
        # self.prompt_id = 'scraping-policy'
        # return prompts[0]['content']

    def load_cache(self):
        """
        Loads the response cache from a JSON file. This method initializes the cache attribute 
        of the GPT class by attempting to read from a specified file path. If the file does 
        not exist, it sets the cache to an empty dictionary, effectively starting with no 
        cached data.
        """
        try:
            with open(self.cache_file_path, 'r') as file:
                self.cache = json.load(file)
        except FileNotFoundError:
            self.cache = {}

    def save_cache(self):
        """
        Saves the current state of the cache to a JSON file. This method writes the contents 
        of the `cache` attribute to a file specified by `cache_file_path`. The JSON data is 
        formatted with an indentation of 4 spaces, making it human-readable.
        """
        
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)

        with open(self.cache_file_path, 'w') as file:
            json.dump(self.cache, file, indent=4, ensure_ascii=False)

    def count_tokens(self, text):
        if self.disable_token_counting:
            return 0
        else:
            # https://github.com/openai/tiktoken/blob/main/tiktoken/model.py#L87
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except:
                print(f"[Warning] No tokenizer found for model '{self.model}', using 'o200k_base' as fallback.")
                encoding = tiktoken.get_encoding("o200k_base")
            token_count = len(encoding.encode(text))
            return token_count

    @staticmethod
    def stringify_messages(messages_list):
        """
        Converts a list of message dictionaries into a single string representation.
        
        Args:
            messages (list): A list of dicts with 'role' and 'content' keys.
            
        Returns:
            str: A single string representing the full conversation.
        """
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_list])

    def make_openai_request(self, final_prompt=None, messages_list=None):
        """
        Makes a request to the OpenAI Chat API to generate completions for a given prompt.

        Parameters:
        - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.

        Returns:
        - str: The response from the OpenAI Chat API containing the completion for the given prompt.
        """
        if final_prompt is not None:
            messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    # {"role": "user", "content": self.USER_PROMPT_1},
                    # {"role": "assistant", "content": self.ASSISTANT_PROMPT_1},
                    {"role": "user", "content": final_prompt}
                ]
        elif messages_list is not None:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT},] + messages_list
        else:
            raise ValueError("Either final_prompt or messages_list must be provided.")
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,           # lower temperature for more deterministic outputs
            top_p=self.top_p,                 # lower top_p to decrease randomness
            messages=messages
        )
        return response['choices'][0]['message']['content'].strip(" \n")

    async def make_openai_request_async(self, session, final_prompt=None, messages_list=None):
        """
        Makes an asynchronous request to the OpenAI Chat API to generate completions for a given prompt.

        Parameters:
        - session (aiohttp.ClientSession): A session object used to make asynchronous requests.
        - final_prompt (str): The final prompt to be sent to the OpenAI Chat API.

        Returns:
        - str: The response from the OpenAI Chat API containing the completion for the given prompt.
        """
        url = self.base_url  # endpoint for gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if final_prompt is not None:
            messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    # {"role": "user", "content": self.USER_PROMPT_1},
                    # {"role": "assistant", "content": self.ASSISTANT_PROMPT_1},
                    {"role": "user", "content": final_prompt}
                ]
        elif messages_list is not None:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT},] + messages_list
        else:
            raise ValueError("Either final_prompt or messages_list must be provided.")
        payload = {
            "model": self.model,
            "temperature": self.temperature,           # lower temperature for more deterministic outputs
            "top_p": self.top_p,                 # lower top_p to decrease randomness
            "messages": messages
        }
        input_token_count = self.count_tokens(final_prompt)
        self.token_usage["input_tokens"] += input_token_count  # Track input tokens


        async with session.post(url, json=payload, headers=headers) as response:
            # print(f"status code: {response.status}")
            # status code of 200 indicates a successful connection
            if response.status == 200:
                data = await response.json()
                output_text = data['choices'][0]['message']['content'].strip(" \n")
                output_token_count = self.count_tokens(output_text)
                self.token_usage["output_tokens"] += output_token_count
                return output_text
            else:
                return None


    async def process_batch_async(self, session, batch, custom_guidelines_prompt=None):
        """
        Processes a single batch of prompts asynchronously, leveraging caching to optimize API usage. 
        Before making an API request, the method checks if a response for the formatted prompt is already 
        stored in the cache. If found, it uses the cached response; otherwise, it sends a request to the 
        OpenAI API.

        Parameters:
        - session (aiohttp.ClientSession): A session object used for making asynchronous HTTP requests.
        - batch (list of str): A list of original prompts to be processed in the batch.
        - custom_guidelines_prompt (str, optional): A custom prompt template that can be formatted with the 
          original prompt. If provided, it overrides the default guidelines prompt template for this batch.

        Returns:
        - list of str: A list of responses from the OpenAI Chat API. Each response corresponds to a prompt in 
          the batch. Responses are retrieved from the cache if available; otherwise, they are fetched from the API.
        """
        responses = []
        tasks = []

        for formatted_prompt in batch:
            # formatted_prompt = custom_guidelines_prompt.format(prompt) if custom_guidelines_prompt else self.GUIDELINES_PROMPT_TEMPLATE.format(prompt)
            # cache_id = f"{self.prompt_id}: {formatted_prompt}"
            is_message_list = False
            if isinstance(formatted_prompt, list):
                # If the prompt is a list, convert it to a string representation
                prompt_id = self.stringify_messages(formatted_prompt)
                is_message_list = True
            else:
                # TODO: hash it?
                prompt_id = formatted_prompt
            input_token_count = self.count_tokens(formatted_prompt)

            if prompt_id in self.cache:
                self.token_usage["cached_input_tokens"] += input_token_count  # Cached input tokens
                responses.append(self.cache[prompt_id])
                print(f"Using cached response for prompt ID: {prompt_id}")
            else:
                self.token_usage["input_tokens"] += input_token_count  # Normal input tokens
                if is_message_list:
                    task = asyncio.create_task(self.make_openai_request_async(session, messages_list=formatted_prompt))
                else:
                    task = asyncio.create_task(self.make_openai_request_async(session, formatted_prompt=formatted_prompt))
                tasks.append((prompt_id, task))

        api_responses = await asyncio.gather(*(task[1] for task in tasks))

        for (prompt_id, _), response in zip(tasks, api_responses):
            if response:
                self.cache[prompt_id] = response
                responses.append(response)
        self.save_cache()

        return responses
    
    def estimate_cost(self):
        """
        Determines which cost estimation to use based on the model and prints cumulative usage.
        """
        model = self.model.lower()
        if 'gpt-4o' in model:
            self._estimate_cost_4o()
        elif 'gpt-3.5' in model or 'o3-mini' in model:
            self._estimate_cost_o3()
        else:
            print(f"No cost estimator defined for model: {self.model}")
    
    def _estimate_cost_4o(self):
        input_cost = (self.token_usage["input_tokens"] / 1_000_000) * 2.50
        cached_input_cost = (self.token_usage["cached_input_tokens"] / 1_000_000) * 1.25
        output_cost = (self.token_usage["output_tokens"] / 1_000_000) * 10.00
        total_cost = input_cost + cached_input_cost + output_cost

        print("\n-----------Estimated Cost (gpt-4o)-----------")
        print(f"  Input Cost: ${input_cost:.4f}")
        print(f"  Cached Input Cost: ${cached_input_cost:.4f}")
        print(f"  Output Cost: ${output_cost:.4f}")
        print(f"  Total Cost: ${total_cost:.4f}\n")

    def _estimate_cost_o3(self):
        input_cost = (self.token_usage["input_tokens"] / 1_000_000) * 1.10
        cached_input_cost = (self.token_usage["cached_input_tokens"] / 1_000_000) * 0.55
        output_cost = (self.token_usage["output_tokens"] / 1_000_000) * 4.40
        total_cost = input_cost + cached_input_cost + output_cost

        print("\n-----------Estimated Cost (gpt-o3-mini)-----------")
        print(f"  Input Cost: ${input_cost:.4f}")
        print(f"  Cached Input Cost: ${cached_input_cost:.4f}")
        print(f"  Output Cost: ${output_cost:.4f}")
        print(f"  Total Cost: ${total_cost:.4f}\n")


    async def process_prompts_in_batches_async(
        self, 
        batch, 
        batch_size=1, 
        parse_func=None,
    ):
        """
        Processes a list of prompts in batches asynchronously by sending them to the OpenAI Chat API.

        Parameters:
        - batch (list of dict): List of dictionaries, each containing 'text' and 'metadata'.
        - batch_size (int, optional): Size of each batch. Defaults to 10.
        - custom_guidelines_prompt (str, optional): Custom guidelines prompt template for formatting.

        Returns:
        - list of dict: List of responses from the OpenAI Chat API, each linked with its metadata.
        """
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(batch), batch_size):
                batch_prompts = batch[i:i + batch_size]
                # batch_prompts = [
                #     item['text'] for item in current_batch  # Remove formatting here
                # ]
                try:
                    batch_responses = await self.process_batch_async(session, batch_prompts)
                except Exception as e:
                    print(f"Skipping batch due to error: {e}")  # Log the error but continue processing
                    continue  # Move to the next batch
                parsed_responses = []
                for response in batch_responses:
                    if parse_func:
                        try:
                            parsed_response = parse_func(response)
                            parsed_response = json.loads(response)
                            parsed_responses.append(parsed_response)
                        except json.JSONDecodeError as e:
                            print("Failed to parse response:", response, "Error:", e)  # debugging output
                            parsed_responses.append(response)  # append the unparsed response if parsing fails
                    else:
                        parsed_responses.append(response)

                # for item, parsed_response in zip(batch_prompts, parsed_responses):
                #     response_with_metadata = {"input": item, "response": parsed_response}
                #     final_responses.append(response_with_metadata)
        self.estimate_cost()
        return parsed_responses


    def clear_cache(self):
        """
        Clears the entire cache, both in-memory and in the file.

        Returns:
        - None
        """
        self.cache = {} 
        self.save_cache()
        