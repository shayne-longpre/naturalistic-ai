import sys
import asyncio
import argparse
import pandas as pd
from helpers import gpt
from utils import *

# Preliminaries
sys.path.append("./")
sys.path.append("src/")


def make_prompt(args):
    PREAMBLE = f"""You are a high-quality annotation assistant. Your task is to annotate conversation logs between users and AI chatbots. You will be given a specific task description, all possible label options for the task, and a part of the conversation, including the user prompt and model response from both previous and current turns. These might be pulled from any part of a multi-turn conversation. As a high-quality annotator you will diligently provide annotations on the current turn that are:
1. Comprehensive: You will list all relevant annotations as the tasks can be multi-class (only one label is true) or multi-label (multiple categories can be true at once). Pay special attention to subtle, or implied properties of the input conversation. 
2. Precise: You must answer as a JSON list of dictionaries of exact labels name(s) and confidence (a score between 0 and 1) without any additional explanation, reasoning, or text.
3. Calibrated: Reflect appropriate confidence in your annotation. If the input is ambiguous or open to multiple interpretations, acknowledge that explicitly.
The conversation log is enclosed between <START_CONVERSATION> and <END_CONVERSATION> tags. The previous conversation turn is enclosed between <START_PREVIOUS_TURN> and <END_PREVIOUS_TURN> tags, while the current conversation turn is enclosed between <START_CURRENT_TURN> and <END_CURRENT_TURN> tags.
The previous conversation turn is provided for context; however, generate an annotation only for the current {args.level_id}. 
Only use information present or inferable from the input. Avoid hallucinations or unjustified assumptions."""

    JSON_INSTRUCTION = f"Return the answer as a JSON list of dictionaries, each with the fields 'labels' (exact label name(s) before ':' and after '- ') and 'confidence' (a score between 0 and 1). Do not include any explanation, reasoning, or additional text."

    TASK_DESCRIPTION = {
        "prompt": {
            "media_format": "Label all types of media that are present in the current user prompt, from the list of options below.",
            "interaction_features": "Label all types of interaction features that are present in the current user prompt, from the list of options below.",
            "topic": "Label all topics that are clearly present in the current user prompt, from the list of options below. This includes the topic of the task (e.g., creative writing would be 'Literature & Writing') and any other topics present in the content of that writing.",
            "multi_turn_relationship": "Label one type of relationship that is present between the previous user prompt and the current user prompt in the conversation, from the list of options below.",
        },
        "response": {
            "media_format": "Label all types of media that are present in the current model response, from the list of options below.",
            "interaction_features": "Label all types of interaction features that are present in the current model response, from the list of options below. Only count these if they are part of the model's communication to the user rather than part of solicited content, like a translation, movie script, or a role it is playing.",
            "answer_form": "Label one type of answer format that represents how the model responds in the current turn, from the list of options below.",
        },
        "turn": {
            "topic": "Label all topics that are present in the current conversation turn, from the list of options below.",
            "sensitive_use_flags": "Label all instances of sensitive content in the current conversation turn, from the list of options below. Please flag any content that relates to these categories, even if it was not inherent in the request: e.g. if the user asks the system to count the words in a violent or sexual story, or if the user asks the system to answer a list of questions (e.g., homework).",
        }
    }

    OPTIONS = {
        "prompt": {
            "media_format": "- Natural language: Text in regular written or spoken language used for communication.\n- Code: Programming scripts, snippets, or syntax in various programming languages, which can be any programming language such as Python, HTML etc.\n- Math/symbols: Mathematical expressions, formulas, or symbolic representations.\n- Formatted enumeration/itemization (bullets/lists): Structured lists of items or points presented with formatting like numbers, bullets, or letters.\n- Charts/Graphs: A visually rendered chart or graphic.\n- Images: Visual representations such as photographs, illustrations, that are not charts/graphs.\n- Audio: Sound files or other auditory content.\n- URLs: Web links that direct users to a specific online resource.\n- Likely retrieved/pasted content: Indicates that the text appears to be copied verbatim from an external source rather than originally composed; it may include quotes, large excerpts, error logs, or reference material paste such as snippets from an external text/code.\n- HTML\n- Other: Any other media type not covered by the above categories, such as 3D models, video, or mixed formats.",
            "interaction_features": "- Role-assignment: The user asks or tells the model to assume a role with specific traits or qualifications.\n- Jailbreak attempt: An apparent attempt from the user to get the system/model to violate its guardrails or usage policy.\n- Courtesy/Politeness: The user uses polite language (“please” or “thank you”) or greetings (“hello” or “good morning”) when interacting with the model.\n- Reinforcement/Praise/Scolding: The user praises the model (or scolds it when it does wrong).\n- Companionship: The user expresses emotional reliance on the model as an emotional being, capable of sustaining a human-like relationship.\n- None: The user does not treat the model like a person in any of the ways mentioned above.",
            "topic": "- No clear task\n- Information retrieval (general info from web): Requesting general information sourced from the web.\n- Information retrieval (general info from prompt content): Requesting information sourced from the prompt itself, such as an accompanying body of text.\n- Content generation (brainstorming / ideation): Requests to generate ideas, suggestions, or options without detailed elaboration.\n- Content generation (creative / fiction writing): Requests to write imaginative or fictional content, including stories or poems.\n- Content generation (academic / essay writing): Requests to produce structured, fact-based content such as essays, reports, or academic discussions.\n- Content generation (administrative writing): Requests to create/write formal or practical documents such as emails, policies, or reports.\n- Content generation (code): Requests to write programming code for a given task or problem.\n- Content generation (code documentation): Requests to provide comments or descriptions for code to help others understand its purpose and functionality.\n- Content generation (prompts for another AI system): Requests to write prompts or inputs for another AI system to ingest.\n- Content generation (general prose, discussion or explanation): Requests to generate general prose about a topic, that is not creative writing, or academic/essay writing.\n- Content generation (other): Requests for content that doesn’t clearly fit into any of the other categories\n- Editorial & formatting (Natural language content editing): Requests to modify the tone, grammar, word choice, style, or content.\n- Editorial & formatting (Natural language style or re-formatting): Requests to change the structure or presentation of text, without significant changes to the content itself.\n- Editorial & formatting (Code content editing): Requests to edit the logic or structure of the code to improve efficiency or functionality.\n- Editorial & formatting (Code style and re-formatting): Requests to edit the style or formatting of the code, without significantly changing the underlying content or purpose. This might include indentations, variable naming, comment usage, or other stylistic conventions.\n- Editorial & formatting (Content summarization): Requests to condense the content to highlight key points while removing less important details.\n- Editorial & formatting (Content expansion): Requests to expand on existing content by adding new details, examples, or explanations to it\n- Editorial & formatting (Information processing & re-formatting): Requests to transform data or information into a new structure or format, such as converting tables to paragraphs or file formats to other file formats.\n- Information analysis (Content explanation / interpretation): Requests to explain or interpret concepts, ideas, or processes.\n- Information analysis (Content quality review or assessment): Requests to evaluate the quality/credibility/relevance of content based on some criteria.\n- Information analysis (Content Classification): Requests to categorize content into predefined groups or categories based on characteristics, subject, or purpose\n- Information analysis (Ranking or Scoring): Requests to assign a numerical or ordinal value to content based on defined parameters.\n- Information analysis (Other content analysis / description): Other types of analysis or providing descriptive insights into the nature, structure, or features of the content\n- Translation (language to language): Requests to convert content from one language to another \n- Role-play / social simulation (platonic companion / friend): Chatting with the model as a friendly, non-romantic companion, offering casual conversations, advice, or companionship without romantic elements\n- Role-play / social simulation (romantic companion): Chatting with the model as a romantic partner for conversations, emotional support, or role-playing scenarios involving love, dating, or sex\n- Role-play / social simulation (simulation of real person / celebrity): Asking the model to mimic the behavior, speech, or personality of a known individual, such as a historical figure or celebrity, for fun or role-playing \n- Role-play / social simulation (user study persona simulations or polling): Asking the model to act as a hypothetical user or persona to simulate responses for user studies, focus groups, or polling, to gather insights or test scenarios, e.g., for research\n- Role-play / social simulation (therapist / coach): Asking the model to act as a therapist or life coach who offers guidance, strategies for self-improvement, or mental health support.\n- Advice, Guidance, & Recommendations (Instructions / How-to): Requests to provide step-by-step instructions or procedures for completing a task or achieving a specified goal.\n- Advice, Guidance, & Recommendations (Social and personal advice): Requests to offer suggestions for navigating social situations, relationships, or personal issues.\n- Advice, Guidance, & Recommendations (Professional advice): Requesting guidance related to career decisions or workplace behavior\n- Advice, Guidance, & Recommendations (Activity / product recommendations): Request to suggest specific activities, experiences, or products based on user needs or preferences.\n- Advice, Guidance, & Recommendations (Action planning (scheduling, robotics)): Requests asking for an action plan or instructions, as usable for robotic scheduling or other automatic tools.\n- Reasoning (Mathematical or numerical problem solving): Requests involving calculations, numerical reasoning, or solving equations or optimization problems.\n- Reasoning (Verbal problems, logic games, puzzles or riddles): Task requests requiring logical reasoning in verbal form such as solving riddles or word-based puzzles\n- Reasoning (Other general problem solving): Other requests requiring reasoning that don't fit into math-based or verbal reasoning tasks as described above\n- Other: Other requests that don’t fit any of the above categories.",
            "multi_turn_relationship": "- First request: The initial input or prompt provided by the user to start a new conversation.\n- Completely new request: A follow-up input from the user that shifts to a different topic or context, unrelated to the previous turn.\n- Re-attempt/revision on prior task\Revision request: The user re-attempts the same task, by revising their prompt in some way, A follow-up input where the user requests the same task or information again, usually because the of indicating the prior response was not fully satisfactory. dissatisfaction or misunderstanding.\n- New variation of prior task (e.g. translate A, now translate B): A follow-up input that is a similar task to the prior turn, but introduces a distinct query. The user is repeating the prompt/task in different ways.\n- Extend, deepen, or build on prior task: This prompt expands, deepens, or builds on the prior task, indicating the response was helpful, but there are additional tasks/questions to now explore.",
        },
        "response": {
            "media_format": "- Natural language: Text in regular written or spoken language used for communication.\n- Code: Programming scripts, snippets, or syntax in various programming languages, which can be any programming language such as Python, HTML etc.\n- Math/symbols: Mathematical expressions, formulas, or symbolic representations.\n- Formatted enumeration/itemization (bullets/lists): Structured lists of items or points presented with formatting like numbers, bullets, or letters.\n- Charts/Graphs: A visually rendered chart or graphic.\n- Images: Visual representations such as photographs, illustrations, that are not charts/graphs.\n- Audio: Sound files or other auditory content.\n- URLs: Web links that direct users to a specific online resource.\n- Likely retrieved/pasted content: Indicates that the text appears to be copied verbatim from an external source rather than originally composed; it may include quotes, large excerpts, error logs, or reference material paste such as snippets from an external text/code.\n- HTML\n- Other: Any other media type not covered by the above categories, such as 3D models, video, or mixed formats.",
            "interaction_features": "- Self-Disclosure: The model explicitly states it is a language model/AI.\n- Content-Direct Response: The model explicitly states it is a person.\n- Content-Preferences/Feelings/Opinions/Religious beliefs: The model expresses “personal” preferences, emotions, opinions, or religious beliefs.\n- Apology: The system apologizes for something.\n- Content-Empathy: The model expresses empathy or human emotions to attempt to relate to a user.\n- Register and Style- Phatic Expressions: The model uses phatic expressions (pleasantries that maintain social relations but do not convey meaningful information.).\n- Register and Style- Expressions of Confidence and Doubt: The model includes hedging expressions to reflect uncertainty in its outputs.\n- Non-Personalization: The model corrects or rejects attempts to personalize it with human attributions, preferences or emotions.\n- None: The model does not include any of the other interaction features.",
            "answer_form": "- Refusal to answer (with explanation): The system declines to respond and provides a reason or justification for the refusal.\n- Refusal to answer (without explanation): The system declines to respond without offering any reasoning for its refusal.\n- Partial refusal, expressing uncertainty, disclaiming: The system provides a partial response while indicating uncertainty or disclaiming full accuracy.\n- Direct Answer / Open Generation: The system provides a direct response or generates an open-ended output.\n- Continuation of Input: The system continues or builds upon the user's input rather than providing a discrete response.\n- Request for Information or Clarification: The system responds by asking the user for more details, clarifications, or specific information needed to address the request accurately.",
        },
        "turn": {
            "topic": "- None\n- Adult & Illicit Content: Content that involves mature themes or topics for adult audiences, including content related to sex, drugs, and alcohol.\n- Art & Design: Topics about artistic techniques, works of art, or design principles.\n- Business & Finances: Topics related to companies, management, financial documents, or personal financial planning.\n- Culture: Topics related to individual tastes, traditions, or societal norms.\n- Economics: Topics related to financial systems, markets, or economic theories.\n- Education: Topics on learning, schools, or basic information.\n- Employment & Hiring: Topics related to job searching, hiring practices, or careers.\n- Entertainment, Hobbies & Leisure: Topics related to movies, music, games, or leisure activities.\n- Fantasy / Fiction / Fanfiction\n- Fashion & Beauty: Topics related to clothing, trends, beauty standards, or beauty tips.\n- Food & Dining: Topics related to cuisine, recipes, or dining experiences.\n- Geography: Topics related to locations, physical features, or geopolitical aspects of the Earth.\n- Health & Medicine: Topics related to physical or mental health, treatments, or medical advice.\n- History: Topics involving historical events, figures, or analysis.\n- Housing: Topics related to real estate, housing policies, or other living arrangements.\n- Immigration / Migration: Topics related to visas, moving across borders, or cultural adaptation.\n- Insurance & Social Scoring: Topics related to insurance policies, social credit systems, or insurance companies.\n- Interpersonal Relationships & Communication: Topics related to relationships, communication skills, or conflict resolution.\n- Law, Criminal Justice, Law Enforcement: Topics related to the law, policing, justice systems, or crime.\n- Lifestyle: Topics related to everyday life, habits, or personal routines.\n- Linguistics & Languages: Topics related to language structures, grammar, or translation.\n- Literature & Writing: Topics about written works, writing techniques, or literary analysis.\n- Math & Sciences: Topics related to mathematics, physics, biology, chemistry, or other scientific fields.\n- Nature & Environment: Topics related to ecosystems, plants, animals, or conservation.\n- News & Current Affairs: Questions or discussions on recent events.\n- Non-software Engineering & Infrastructure: Topics related to mechanical engineering, construction, design, or infrastructure.\n- Politics & Elections: Topics related to political ideologies, political contestants, campaigns, or elections.\n- Psychology, Philosophy & Human Behavior: Topics related to mental processes, philosophical theories, or patterns of human behavior.\n- Religion & Spirituality: Topics related to religious beliefs, practices, or spiritual concepts.\n- Social Issues & Movements: Topics related to societal challenges or activism.\n- Sports: Topics related to sporting events, teams, or exercises.\n- Technology, Software & Computing: Topics on tech advancements, programming, or other software.\n- Transportation: Topics related to vehicles, public transport, or transit systems.\n- Travel & Tourism: Topics related to destinations, itineraries, or travel tips.\n- Video Games: Topics related to video games, video game design, etc.\n- Other: Topics that do not fit into any of the specified categories.",
            "sensitive_use_flags": "- None: Content that doesn’t involve any restricted use.\n- Inciting violence, hateful or other harmful behavior (harassment & bullying): Content that encourages or facilitates harassment and/or bullying behaviors, such as cyberbullying, doxxing, and use of derogatory language.\n- Inciting violence, hateful or other harmful behavior (physical harm): Content that encourages or facilitates behaviors that could lead to physical injury or danger.\n- Inciting violence, hateful or other harmful behavior (self-harm): Content that encourages or facilitates engagement in activities that cause harm to oneself.\n- Criminal planning or other suspected illegal activity not listed elsewhere: Content that details or suggests criminal planning or other suspected illegal activities.\n- Cyberattacks (enabling/enacting malware, computer viruses, worms, malicious code, ...): Content that enables or enacts cyberattacks such as malware, computer viruses, worms, malicious code, etc.\n- Weapons & drugs: Content that promotes, facilitates, or instructs the manufacturing, distribution, or use of weapons or illegal drugs.\n- CBRN-related outputs: Content that involves CBRN-related (Chemical, Biological, Radiological, and Nuclear) information.\n- Sexually explicit content (real person): Content that involves sexually explicit content about a real person.\n- Sexually explicit content (fictitious person): Content that involves sexually explicit content about a fictitious person.\n- Sexually explicit content (Request/discussion of CSAM): Content that involves sexually explicit content classified as Child Sexual Abuse Material (CSAM).\n- Sexually explicit content (Other): Content that involves sexually explicit content in general.\n- Impersonation attempts: Content that involves attempts to mimic or assume the identity of another person or organization, such as creating fake social media profiles or imitating someone’s writing style.\n- Misinformation: Content that involves the generation of inaccurate, manipulative, or unverified information.\n- Privacy concerns (Possible identifiable information (e.g., social security number, driver's license number, passport number, biometric records, financial account or credit card number, personal address, and phone number)): Content that involves information that directly or indirectly identifies a natural (non-fictional and alive) person (e.g., name, an identification number, biometric records, financial account or credit card number, personal address, phone number, demographic information, and other special characteristics which expresses the physical, physiological, genetic, mental, commercial, cultural or social identity of the natural person).\n- Privacy concerns (Possible sensitive information (e.g., API keys, passwords, other confidential information)): Content that involves sensitive information about an individual or an entity, including confidential information such as passwords, user credentials, API keys, proprietary algorithms, and other sensitive information that could lead to harmful outcomes such as physical, economic, reputational, psychological, autonomy, discrimination, and relationship harms.\n- Generating spam: Content that involves the generation of spam, such as bulk emails with promotional or phishing intents, fake product reviews or ratings, content farms, forum spam, and SEO spam.\n- Generating defamatory content: Content that involves the generation of defamatory content targeting individuals or organizations.\n- Output misrepresentation (Expressly and intelligibly disclaiming that the information and/or content is machine generated/attempt to hide AI use or source material): Content that involves or responds to requests to hide the use of AI in content generation tasks, or to hide source material by making alterations to the original piece to make it appear as a new creation.\n- Output misrepresentation (Automated decision-making without disclosure): Content that involves or responds to requests that intend to make automated decisions without disclosure.\n- Discriminatory practices (Misrepresentation, stereotyping, or inappropriate reference to sensitive attributes): Content that involves discriminatory practices, such as misrepresentation, stereotyping, or inappropriate reference to sensitive attributes.\n- Possible presence of copyrighted, unreferenced material: Content that involves possible presence of copyrighted, unreferenced material.\n- Potential violation of external policy / ethics\n- Other: Other sensitive, harmful, or restricted use of LLM in a request or the a response.",
        }
    }
    task_description = TASK_DESCRIPTION[args.level_id][args.prompt_id]
    options = OPTIONS[args.level_id][args.prompt_id]

    prompt = f"""{PREAMBLE}

Task description: {task_description}
Options: {options}

<START_CONVERSATION>

<START_PREVIOUS_TURN>
{{prev_text}}
<END_PREVIOUS_TURN>

<START_CURRENT_TURN>
{{curr_text}}
<END_CURRENT_TURN>

<END_CONVERSATION>

{JSON_INSTRUCTION}
Response: """
    return prompt


def format_conversation_turns(conversation):
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        user_turn = conversation[i]
        assistant_turn = conversation[i + 1] if i + 1 < len(conversation) else None
        if user_turn["role"] == "user" and assistant_turn and assistant_turn["role"] == "assistant":
            pairs.append((user_turn["text"], assistant_turn["text"]))

    formatted_turns = []
    for i in range(len(pairs)):
        # Previous turn
        if i == 0:
            prev_user = "None"
            prev_assistant = "None"
        else:
            prev_user, prev_assistant = pairs[i - 1]

        # Current turn
        curr_user, curr_assistant = pairs[i]

        prev_text = f"Previous user prompt: {prev_user}\nPrevious model response: {prev_assistant}"
        curr_text = f"Current user prompt: {curr_user}\nCurrent model response: {curr_assistant}"

        formatted_turns.append((prev_text, curr_text))
    return formatted_turns


def extract_samples_and_metadata(args, dataframe, existing_ex_ids):
    sample, metadata = [], []

    for _, row in dataframe.iterrows():
        if row["ex_id"] in existing_ex_ids:
            continue

        conversation = row["conversation"]
        formatted_pairs = format_conversation_turns(conversation)

        for prev_text, curr_text in formatted_pairs:
            prompt = make_prompt(args)
            prompt = prompt.replace("{prev_text}", prev_text).replace("{curr_text}", curr_text)

            sample.append(prompt)
            metadata.append({
                "ex_id": row["ex_id"],
                "dataset_id": row["dataset_id"],
                "model": row["model"]
            })
    return sample, metadata


async def run_gpt(args, batch_size=1):
    existing_ex_ids = load_existing_ex_ids(args.save)    
    dataframe = pd.read_json(args.input, orient="records")
    formatted_prompts, metadata = extract_samples_and_metadata(args, dataframe, existing_ex_ids)
    print(f"Formatted prompts: {len(formatted_prompts)}, Metadata: {len(metadata)}")
    
    if not formatted_prompts:
        print("All examples already exist in the save file.")
        return

    gpt_instance = gpt.GPT(model=args.model_id, prompt=args.prompt_id)

    for batch, meta_batch in zip(batch_generator(formatted_prompts, batch_size), batch_generator(metadata, batch_size)):
        try:
            batch_responses = await gpt_instance.process_prompts_in_batches_async(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        
        batch_output = []
        print("Prompt: ", batch[0])
        for response, meta in zip(batch_responses, meta_batch):
            print("Response: ", response)
            response_entry = {
                **meta,
                "model_id": args.model_id,
                "level_id": args.level_id,
                "prompt_id": args.prompt_id,
                "input": batch[0],
                "response": response
            }
            batch_output.append(response_entry)

            if batch_output:
                append_jsonl(batch_output, args.save)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input json file.")
    parser.add_argument("--level_id", required=True, default=None)
    parser.add_argument("--prompt_id", required=True, default=None)
    parser.add_argument("--model_id", required=True, default=None, choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o3-mini"])
    parser.add_argument("--save", type=str, required=True, help="Save path.")
    args = parser.parse_args()

    asyncio.run(run_gpt(args))
