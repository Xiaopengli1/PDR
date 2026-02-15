"""
PDR Profile Extraction: Build structured user profiles from heterogeneous sources.
Transforms drafts, notes, and documents into JSON profiles for downstream personalization.
"""
import os
import re
import json
from typing import List, Union

from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader import PDFLoader, JsonFileLoader, TextLoader
from deepsearcher.tools import log

PERSON_UNDERSTAND_PROMPT = """You are an intelligent personal-assistant AI.

Task  
1. You will receive a “personal information document” (unstructured or semi-structured text).  
2. Analyse its content thoroughly.  
3. Produce **only** a JSON object with the fields listed below.  
4. If a piece of information is missing or ambiguous, use logical in-context inference.

Required JSON fields  
- Name  
- Age  
- Gender  
- ProfessionalBackground  
- SkillsAndCompetencies  
- EducationAndCertifications  
- ProjectsAndAchievements  
- InterestsAndHobbies  
- Languages  
- PersonalityTraits  
- CareerGoalsAndAspirations  
- PreferredWorkEnvironment  
- LocationAndMobility  
- NetworkingAndAffiliations  
- VolunteeringAndExtracurriculars  
- ToolsAndTechnologies  
- LearningInterests  
- CreativeOutlets  
- FitnessAndWellbeing  
- ReadingAndMediaConsumption  
- ProductivityMethods  
- ValuesAndBeliefs  
- SocialPresence  
- ResponseGenerationAnalysis  

Input:
{content}

Output:  
Put you reasoning process in <think> </think> and put the final output in <output> </output>
"""

def save_metadata(metadata: dict, save_path: str = 'file_snapshot.json') -> None:
    # pass
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(snapshot_path: str = 'file_snapshot.json') -> dict:
    if not os.path.exists(snapshot_path):
        return {}
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_modified_files(folder_path: str, snapshot_path: str = 'file_snapshot.json') -> str:


    contents = []
    current_metadata = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(('.txt', '.md', 'json', 'jsonl')):
                    text_loader = TextLoader()
                    content = text_loader.load_file(file_path).__str__()
                elif file.endswith('.pdf'):
                    pdf_loader = PDFLoader()
                    content = pdf_loader.load_file(file_path).__str__()
                contents.append(content)
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")

    return "\n\n".join(contents)


def save_json(data: object, output_path: str = 'personalized_summary.json') -> None:
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        log.color_print(f"✅ JSON saved to {output_path}")
    except (json.JSONDecodeError, TypeError) as e:
        log.color_print(f"❌ Failed to save JSON: {e}")


def personalized_understanding(paths_or_directory: Union[str, List[str]], llm: BaseLLM):

    summary_path = os.path.join(paths_or_directory, "personalized_summary.json")

    if os.path.exists(summary_path):
        try:
            # Try to read existing file content
            with open(summary_path, 'r', encoding='utf-8') as file:
                parsed_response = json.load(file)
            
            # Build mock response
            mock_response = f"<output>{json.dumps(parsed_response)}</output>"
            log.color_print(f"⏩ Using cached summary: {summary_path}")
            log.color_print(f"<personal summary> {parsed_response} </personal summary>\n")
            
            # Return mock result (token consumption set to 0)
            return mock_response, 0
        except Exception as e:
            log.color_print(f"⚠️ Summary file corrupted, regenerating: {str(e)}")

    all_text = read_modified_files(paths_or_directory)

    chat_response = llm.chat(
        messages=[{
            "role": "user",
            "content": PERSON_UNDERSTAND_PROMPT.format(content=all_text[:100000])
        }]
    )

    response_content = chat_response.content

    think_pattern = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
    output_pattern = re.search(r'<output>(.*?)</output>', response_content, re.DOTALL)

    # Extract matched content
    think_content = think_pattern.group(1).strip() if think_pattern else None
    output_content = output_pattern.group(1).strip() if output_pattern else None


    print(output_content)

    parsed_response, _ = llm.literal_eval(output_content)



    # total_token = chat_response.total_tokens
    total_token = 0

    save_json(data=parsed_response, output_path = summary_path )
    if think_content:
        log.color_print(f"<think> {think_content} </think>\n")
        log.color_print(f"<personal summary> {parsed_response} </personal summary>\n")
    else:
        log.color_print(f"<personal summary> Personalized Summary: {parsed_response} </personal summary>\n")

    return response_content, total_token


