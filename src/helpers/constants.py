"""Constants for the helpers module."""


SYSTEM_PROMPTS_FPATH = "data/static/system_prompts.json"
DETAILED_SYSTEM_PROMPTS_FPATH = "data/static/detailed_system_prompts.json"


FUNCTION_ANNOTATION_LABEL_ABBREVIATIONS = {
    "information retrieval: general info from web": "IR: web",
    "information retrieval: general info from prompt content": "IR: prompt",
    "content generation: brainstorming / ideation": "Gen: ideas",
    "content generation: creative / fiction writing": "Gen: creative",
    "content generation: non-fiction / academic / essay writing": "Gen: academic",
    "content generation: administrative writing": "Gen: admin",
    "content generation: code": "Gen: code",
    "content generation: code documentation": "Gen: code doc",
    "content generation: prompts for another ai system": "Gen: prompts",
    "content generation: general prose, discussion or explanation": "Gen: prose",
    "content generation: other": "Gen: other",
    "editorial & formatting: natural language content editing": "E&F: NL edit",
    "editorial & formatting: natural language style or re-formatting": "E&F: NL style",
    "editorial & formatting: code content editing": "E&F: code edit",
    "editorial & formatting: code style and re-formatting": "E&F: code style",
    "editorial & formatting: content summarization": "E&F: summarize",
    "editorial & formatting: content expansion": "E&F: expand",
    "editorial & formatting: information processing & re-formatting": "E&F: reformat",
    "information analysis: content explanation / interpretation": "Analysis: explain",
    "information analysis: content quality review or assessment": "Analysis: review",
    "information analysis: content classification": "Analysis: class",
    "information analysis: ranking or scoring": "Analysis: rank",
    "information analysis: other content analysis / description": "Analysis: other",
    "translation (language to language)": "Translate",
    "role-play / social simulation: platonic companion / friend": "RP: friend",
    "role-play / social simulation: romantic companion": "RP: romantic",
    "role-play / social simulation: simulation of real person / celebrity": "RP: celebrity",
    "role-play / social simulation: user study persona simulations or polling": "RP: persona",
    "role-play / social simulation: therapist / coach": "RP: therapist",
    "advice, guidance, & recommendations: instructions / how-to": "Advice: how-to",
    "advice, guidance, & recommendations: social and personal advice": "Advice: personal",
    "advice, guidance, & recommendations: professional advice": "Advice: prof",
    "advice, guidance, & recommendations: activity / product recommendations": "Advice: product",
    "advice, guidance, & recommendations: action planning (scheduling, robotics)": "Advice: planning",
    "reasoning: mathematical or numerical problem solving": "Reason: math",
    "reasoning: verbal problems, logic games, puzzles or riddles": "Reason: logic",
    "reasoning: other general problem solving": "Reason: general",
    "no clear task": "Unclear",
    "other": "Other"
}


ANNOTATION_TAXONOMY_REMAPPER = {
    'prompt_multi_turn_relationship': {
        'First Request': [
            'first request',
            'first request the initial input or prompt provided by the user to start a new conversation',
        ],
        'New Variation Of Prior Task': ['new variation of prior task'],
        'Re-Attempt / Revision On Prior Task': ['re attempt revision on prior task', 're attempt revision on prior request'],
        'Extend, Deepen, Or Build On Prior Task': ['extend deepen or build on prior task'],
        'Completely New Request': ['completely new request'],
    },
    'prompt_media_format': {
        'Code': [
            'code',
            'code programming scripts snippets or syntax',
            'code programming scripts snippets or syntax in various programming languages',
            'code programming scripts snippets or syntax in various programming languages which can be any programming language such as python html etc',
        ],
        'Charts / Graphs': ['charts graphs'],
        'Natural Language': [
            'natural language',
            'natural language text in regular written or spoken language used for communication',
            'natural language text',
        ],
        'Images': ['images'],
        'Other': ['other'],
        'Formatted Enumeration / Itemization': [
            'formatted enumeration itemization structured lists',
            'formatted enumeration itemization',
            'formatted enumeration itemization bullets lists',
            'formatted enumeration itemizatio',
            'formatted enumeration itemization bullets lists structured lists of items or points presented with formatting like numbers bullets or letters',
            'formatted enumeration itemization structured lists of items or points presented with formatting like numbers bullets or letters e g bullets lists',
            'formatted enumeration itemization structured lists of items or points presented with formatting like numbers bullets or letters',
        ],
        'Math / Symbols': ['math symbols'],
        'URLs': ['urls', 'url'],
        'Likely Retrieved / Pasted Content': [
            'likely retrieved pasted content',
            'likely retrieved pasted content indicates that the text appears to be copied verbatim from an external source rather than originally composed it may include quotes large excerpts error logs or reference material paste such as snippets from an external text code',
        ],
        'HTML': ['html'],
        'Audio': ['audio'],
    },
    'response_media_format': {
        'Code': [
            'code',
            'code programming scripts snippets or syntax',
            'code programming scripts snippets or syntax in various programming languages',
            'code programming scripts snippets or syntax in various programming languages which can be any programming language such as python html etc',
        ],
        'Charts / Graphs': ['charts graphs'],
        'Natural Language': [
            'natural language', 'natural language text in regular written or spoken language used for communication', 'natural language text',
        ],
        'Images': ['images'],
        'Other': ['other'],
        'Formatted Enumeration / Itemization': [
            'formatted enumeration itemization structured lists',
            'formatted enumeration itemization',
            'formatted enumeration itemization bullets lists',
            'formatted enumeration itemizatio',
            'formatted enumeration itemization bullets lists structured lists of items or points presented with formatting like numbers bullets or letters',
            'formatted enumeration itemization structured lists of items or points presented with formatting like numbers bullets or letters e g bullets lists',
            'formatted enumeration itemization structured lists of items or points presented with formatting like numbers bullets or letters',
        ],
        'Math / Symbols': ['math symbols'],
        'URLs': ['urls', 'url'],
        'Likely Retrieved / Pasted Content': [
            'likely retrieved pasted content',
            'likely retrieved pasted content indicates that the text appears to be copied verbatim from an external source rather than originally composed it may include quotes large excerpts error logs or reference material paste such as snippets from an external text code',
        ],
        'HTML': ['html'],
        'Audio': ['audio'],
    },
    "response_interaction_features": {
        "Content: Direct Response": ["content direct response"],
        "Self-Disclosure": ["self disclosure"],
        "Non-Personalization": ["non personalization"],
        "None": [
            "none"
        ],
        "Apology": [
            "apology"
        ],
        "Content: Preferences / Feelings / Opinions / Religious Beliefs": [
            "content preferences feelings opinions religious beliefs"
        ],
        "Register And Style: Expressions Of Confidence And Doubt": [
            "register and style expressions of confidence and doubt",
        ],
        "Register And Style: Phatic Expressions": [
            "register and style phatic expressions"
        ],
        "Content: Empathy": [
            "content empathy",
        ]
    },
    "prompt_function_purpose": {
        "Other": [
            "other"
        ],
        "No Clear Ask": [
            "no clear ask", "no clear task", "none",
        ],
        "Content Generation: Academic / Essay": [
            "content generation academic essay",
            "content generation academic essay writing"
        ],
        "Content Generation: Administrative Writing": [
            "content generation administrative writing"
        ],
        "Content Generation: Brainstorming / Ideation": [
            "content generation brainstorming ideation"
        ],
        "Content Generation: Code": [
            "content generation code",
        ],
        "Content Generation: Code Documentation": [
            "content generation code documentation",
        ],
        "Content Generation: Creative / Fiction Writing": [
            "content generation creative fiction writing",
        ],
        "Content Generation: General Prose, Discussion Or Explanation": [
            "content generation general prose discussion or explanation",
        ],
        "Content Generation: Other": [
            "content generation other",
        ],
        "Content Generation: Prompts For Another AI System": [
            "content generation prompts for another ai system",
        ],
        "Editorial & Formatting: Code Content Editing": [
            "editorial formatting code content editing",
        ],
        "Editorial & Formatting: Code Style And Re-Formatting": [
            "editorial formatting code style and re formatting",
        ],
        "Editorial & Formatting: Content Expansion": [
            "editorial formatting content expansion",
            "content expansion",
            "content generation content expansion"
        ],
        "Editorial & Formatting: Content Summarization": [
            "editorial formatting content summarization",
        ],
        "Editorial & Formatting: Information Processing & Re-Formatting": [
            "editorial formatting information processing re formatting",
            "information processing re formatting"
        ],
        "Editorial & Formatting: Natural Language Content Editing": [
            "editorial formatting natural language content editing",
            "editorial formatting content re phrasing re statement",
            "editorial formatting natural language editing",
        ],
        "Editorial & Formatting: Natural Language Style Or Re-Formatting": [
            "editorial formatting natural language style or re formatting",
        ],
        "Information Analysis: Content Explanation / Interpretation": [
            "information analysis content explanation interpretation",
        ],
        "Information Analysis: Content Quality Review Or Assessment": [
            "information analysis content quality review or assessment",
            "information analysis ranking or scoring",
        ],
        "Information Analysis: Other Content Analysis / Description": [
            "information analysis other content analysis description",
            "information analysis other general problem solving"
        ],
        "Information Analysis: Content Classification": [
            "information analysis content classification"
        ],
        "Information Retrieval: General Info From Prompt Content": [
            "information retrieval general info from prompt content",
        ],
        "Information Retrieval: General Info From Web": [
            "information retrieval general info from web",
            "information retrieval general info from the web",
        ],
        "Reasoning: Mathematical Or Numerical Problem Solving": [
            "reasoning mathematical or numerical problem solving",
        ],
        "Reasoning: Other General Problem Solving": [
            "reasoning other general problem solving",
        ],
        "Reasoning: Verbal Problems, Logic Games, Puzzles Or Riddles": [
            "reasoning verbal problems logic games puzzles or riddles",
        ],
        "Translation (Language To Language)": [
            "translation language to language",
            "translation"
        ],
        "Advice, Guidance, & Recommendations: Social And Personal Advice": [
            "advice guidance recommendations social and personal advice",
        ],
        "Advice, Guidance, & Recommendations: Professional Advice": [
            "advice guidance recommendations professional advice",
        ],
        "Advice, Guidance, & Recommendations: Instructions / How-To": [
            "advice guidance recommendations instructions how to",
        ],
        "Advice, Guidance, & Recommendations: Activity / Product Recommendations": [
            "advice guidance recommendations activity product recommendations",
            "activity product recommendations"
        ],
        "Advice, Guidance, & Recommendations: Action Planning (Scheduling, Robotics)": [
            "advice guidance recommendations action planning scheduling robotics",
        ],
        "Role-Play / Social Simulation: Therapist / Coach": [
            "role play social simulation therapist coach"
        ],
        "Role-Play / Social Simulation: Simulation Of Real Person / Celebrity": [
            "role play social simulation simulation of real person celebrity",
        ],
        "Role-Play / Social Simulation: Platonic Companion / Friend": [
            "role play social simulation platonic companion friend"
        ],
        "Role-Play / Social Simulation: User Study Persona Simulations Or Polling": [
            "role play social simulation user study persona simulations or polling",
        ]
    },
    "turn_topic": {
        'Other': ['other', 'others'],
        'Lifestyle': ['lifestyle'],
        'Economics': ['economics'],
        'Food & Dining': ['food dining'],
        'Immigration / Migration': ['immigration migration'],
        'Law, Criminal Justice, Law Enforcement': [
            'law criminal justice law enforcement'],
        'Art & Design': ['art design'],
        'Interpersonal Relationships & Communication': [
            'interpersonal relationships communication', 'media communication'],
        'Video Games': ['video games'],
        'History': ['history'],
        'Politics & Elections': ['politics elections'],
        'Culture': ['culture'],
        'Health & Medicine': ['health medicine'],
        'Non-software Engineering & Infrastructure': [
            'non software engineering infrastructure', 'engineering infrastructure'],
        'Education': ['education'],
        'Technology, Software & Computing': [
            'technology software computing'],
        'Literature & Writing': ['literature writing'],
        'None': ['none'],
        'Psychology, Philosophy & Human Behavior': [
            'psychology philosophy human behavior'],
        'Linguistics & Languages': ['linguistics languages'],
        'Nature & Environment': ['nature environment', 'agriculture'],
        'Employment & Hiring': ['employment hiring'],
        'Math & Sciences': ['math sciences', 'science mathematics'],
        'Business & Finances': ['business finances', 'marketing advertising'],
        'Travel & Tourism': ['travel tourism'],
        'Entertainment, Hobbies & Leisure': [
            'entertainment hobbies leisure'],
        'Transportation': ['transportation'],
        'Fantasy / Fiction / Fanfiction': ['fantasy fiction fanfiction'],
        'Fashion & Beauty': ['fashion beauty'],
        'News & Current Affairs': ['news current affairs'],
        'Religion & Spirituality': ['religion spirituality'],
        'Sports': ['sports'],
        'Adult & Illicit Content': ['adult illicit content'],
        'Social Issues & Movements': ['social issues movements'],
        'Geography': ['geography'],
        'Insurance & Social Scoring': ['insurance social scoring'],
    },
    "turn_sensitive_use_flags": {
        'Violent, Hateful or Other Harmful Behavior: Physical Harm': [
            'violent hateful or other harmful behavior physical harm',
            'inciting violence hateful or other harmful behavior physical harm'],
        'Sexually Explicit Content: Fictitious Person': [
            'sexually explicit content fictitious person'],
        'None': ['none'],
        'Other': ['other'],
        'Private Information: Possible Sensitive Information': [
            'private information possible sensitive information e g api keys passwords other confidential information'],
        'Discriminatory Practices: Misrepresentation, Stereotyping, or Inappropriate Reference to Sensitive Attributes': [
            'discriminatory practices misrepresentation stereotyping or inappropriate reference to sensitive attributes',
            'discriminatory practices misrepresnetation stereotyping or inappropriate reference to sensitive attributes'],
        'Private Information: Possible Identifiable Information': [
            'private information possible identifiable information e g social security driver s license passport biometric financial address phone number etc'],
        'Possible Presence of Copyrighted, Unreferenced Material': [
            'possible presence of copyrighted unreferenced material'],
        'Uses that May Violate Academic or Professional Standards': [
            'uses that may violate academic or professional standards'],
        'Misinformation': ['misinformation'],
        'Privacy Concerns Possible Identifiable Information': [
            'privacy concerns possible identifiable information'],
        'Output Misrepresentation Expressly Disclaiming AI Use or Source Material': [
            'output misrepresentation expressly and intelligibly disclaiming that the information and or content is machine generated attempt to hide ai use or source material'],
        'Privacy Concerns Possible Sensitive Information': [
            'privacy concerns possible sensitive information', 'privacy concerns possible sensitive information e g api keys passwords other confidential information'],
        'Inciting Violence, Hateful or Other Harmful Behavior Self-Harm': [
            'inciting violence hateful or other harmful behavior self harm'],
        'Output Misrepresentation Automated Decision-Making Without Disclosure': [
            'output misrepresentation automated decision making without disclosure'],
        'Generating Defamatory Content': [
            'generating defamatory content'],
        'Inciting Violence, Hateful or Other Harmful Behavior Harassment Bullying': [
            'inciting violence hateful or other harmful behavior harassment bullying'],
        'Sexually Explicit Content Request Discussion of CSAM': [
            'sexually explicit content request discussion of csam'],
        'Criminal Planning or Other Suspected Illegal Activity': [
            'criminal planning or other suspected illegal activity not listed elsewhere',
            'content related to criminal planning or other suspected illegal activity not listed elsewhere',
        ],
        'Potential Violation of External Policy Ethics': [
            'potential violation of external policy ethics'],
        'Cyberattacks': [
            'cyberattacks enabling enacting malware computer viruses worms malicious code',
            'content related to cyberattacks enabling enacting malware viruses worms etc',
        ],
        'Weapons Drugs': [
            'weapons drugs'],
        'Apparent attempt to impersonate a real person or organization': [
            'impersonation attempts', 'apparent attempt to impersonate a real person or organization',
        ],
        'Sexually Explicit Content Other': [
            'sexually explicit content other'],
    },
    'response_answer_form': {
        'Partial Refusal Expressing Uncertainty Disclaiming': [
            'partial refusal expressing uncertainty disclaiming'],
        'Direct Answer Open Generation': ['direct answer open generation'],
        'Request for Information or Clarification': [
            'request for information or clarification'],
        'Refusal to Answer Without Explanation': [
            'refusal to answer without explanation'],
        'Continuation of Input': ['continuation of input'],
        'Refusal to Answer With Explanation': [
            'refusal to answer with explanation'],
        # "None": ["none"],
    },
    "prompt_interaction_features": {
        'Courtesy Politeness': ['courtesy politeness'],
        'Jailbreak Attempt': ['jailbreak attempt'],
        'None': ['none'],
        'Role-assignment': ['role assignment'],
        'Reinforcement Praise Scolding': ['reinforcement praise scolding', 'reinforcement praise'],
        'Companionship': ['companionship'],
    }
}


ANNOTATION_TAXONOMY_REVERSE_REMAPPER = {
    taskname: {
        label: canonical_label
        for canonical_label, label_list in canonical_label_to_list.items()
        for label in label_list
    }
    for taskname, canonical_label_to_list in ANNOTATION_TAXONOMY_REMAPPER.items()
}
