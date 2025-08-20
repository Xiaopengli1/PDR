import asyncio
import re
from typing import List, Tuple, Any

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results
from deepsearcher.agent.web_searcher import create_search_api


SYSTEM_PROMPT = """You are interacting with the user. Below is his/her personalized information. Use this information to complete the following task in a way that aligns with their unique profile and preferences.

Personalized Information:{personalized_info}"""

SUB_QUERY_PROMPT_v2 = """Please break down the original question into up to four sub-questions based on the content of the question and the user description provided. If the question is simple enough that no decomposition is necessary, keep only the original question.

INPUT:
Original Question: {original_query}

OUTPUT:
<think> Detailed thinking process </think>
<output> A python code list of str of sub-queries </output>
"""

SUB_QUERY_PROMPT_v1 = """To answer this question more comprehensively, please break down the original question into up to four sub-questions according the question content and provided user description.
If this is a very simple question and no decomposition is necessary, then keep the only one original question.


Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Put you reasoning process in <think> </think> and put the response of python code list of str format in <output> </output>
"""

PERSONALIZED_INFO_FIND_PROMPT = """Given the user query below and a list of personalized information categories, identify which categories are most relevant to answering the query.

Query Question: {query}
Personalized Information index: [
    "Name",
    "Age",
    "Gender",
    "ProfessionalBackground",
    "SkillsAndCompetencies",
    "EducationAndCertifications",
    "ProjectsAndAchievements",
    "InterestsAndHobbies",
    "Languages",
    "PersonalityTraits",
    "CareerGoalsAndAspirations",
    "PreferredWorkEnvironment",
    "LocationAndMobility",
    "NetworkingAndAffiliations",
    "VolunteeringAndExtracurriculars",
    "ToolsAndTechnologies",
    "LearningInterests",
    "CreativeOutlets",
    "FitnessAndWellbeing",
    "ReadingAndMediaConsumption",
    "ProductivityMethods",
    "ValuesAndBeliefs",
    "SocialPresence",
    "ResponseGenerationAnalysis"
]

Provide your response in a python code list of str format:"""

SEARCH_EVALUATION_PROMPT = """You are an expert reasoning agent tasked with evaluating whether a set of retrieved text chunks sufficiently answers a user query. Your objective is to use logical thinking and content analysis to determine if the information provided is complete and directly answers the user's intent.

INPUT FORMAT::
Query: {query}
Retrieved Chunks: {chunks}

OUTPUT FORMAT:
<think>Step-by-step reasoning: analyze the relevance, completeness, and clarity of the retrieved chunks in relation to the query.</think>

<output>
"YES" — if the information is complete and answers the query clearly.  
"NO" — if the information is incomplete, ambiguous, or off-topic.
</output>

<revised query>
["If <output> is NO, provide an improved version of one query in this Python list format, focusing on clarity, specificity, or keywords that would improve retrieval."]
</revised query>

Behavioral Rules:
* Think aloud in the <think> section, clearly walking through your reasoning process.
* Only answer "YES" or "NO" in <output>.
* If your answer is "YES", leave <revised query> blank as: [].
* If your answer is "NO", provide ONLY one revised queries as helpful alternatives in the <revised query> section."""

RERANK_PROMPT = """Based on the query questions and the retrieved chunk, to determine whether the chunk is helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Put you reasoning process in <think> </think> and put the response in <output> </output>, you can ONLY return "YES" or "NO".
"""

RERANK_WEB_PROMPT = """Based on the query questions and the retrieved chunk, please determine whether the chunk is little bit helpful in answering any of the query question, you can only return "YES" or "NO", without any other information.

Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

Put you reasoning process in <think> </think> and put the response in <output> </output>, you can ONLY return "YES" or "NO".
"""

REFLECT_PROMPT = """Determine if additional search queries are needed based on the following inputs: the original query, previous sub-queries, and the relevant document chunks retrieved so far. If further research is needed, provide a Python list containing up to 3 new search queries. If no further research is necessary, return an empty list.

INPUT:
Original Query: {question}
Previous Sub Queries: {mini_questions}
Related Chunks: {mini_chunk_str}

OUTPUT:
<output> [New search queries as a list of strings] </output>
<think> [Explain the reasoning behind your decision] </think>

The output should include:
- A Python list of new queries if additional searches are necessary (max 3 queries).
- A reasoning process for the decision in <think> </think>.
- If no further research is needed, return an empty list in <output> </output>."""

SUMMARY_PROMPT = """You are an AI content analysis expert with advanced skills in synthesizing and summarizing complex information, providing clear, precise, and insightful responses that align with the user’s intent. Your goal is to create a well-structured report or summary based on the given context.
Enclose your detailed thought process within <think></think> tags. Be sure to follow these steps:

<think>
1. **Understand the user’s main objective**: Comprehensively analyze the **original query** to identify the goal, expected output, and any preferences the user has shared (e.g., style, format, length).
2. **Account for previous context**: Review **previous sub-queries** and identify patterns or connections with **relevant document chunks** gathered from local and web searches.
3. **Extract and integrate key data**: Combine findings from previous steps, ensuring that you incorporate relevant insights and data that directly address the user’s query.
4. **Structure and refine your response**: Organize your insights into a cohesive, well-organized summary that clearly conveys the findings while ensuring clarity and relevance to the user’s needs.
</think>

Original Query: {question}  
Previous Sub-Queries: {mini_questions}  
Related Chunks: {mini_chunk_str}
"""


# Define helper functions for prompt creation
def create_sub_query_prompt(original_query: str, version: int = 2) -> str:
    if version == 2:
        return f"""Please break down the original question into up to four sub-questions based on the content of the question and the user description provided. If the question is simple enough that no decomposition is necessary, keep only the original question.

INPUT:
Original Question: {original_query}

OUTPUT:
<think> Detailed thinking process </think>
<output> A python code list of str of sub-queries </output>
"""
    else:
        return f"""To answer this question more comprehensively, please break down the original question into up to four sub-questions according the question content and provided user description.
If this is a very simple question and no decomposition is necessary, then keep the only one original question.


Original Question: {original_query}

<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Put you reasoning process in <think> </think> and put the response of python code list of str format in <output> </output>
"""

def extract_think_output(content: str) -> Tuple[str, str]:
    think_pattern = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    output_pattern = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
    think_content = think_pattern.group(1).strip() if think_pattern else None
    output_content = output_pattern.group(1).strip() if output_pattern else None
    return think_content, output_content

def parse_response(output_content: str) -> Any:
    return eval(output_content)  # assuming it's safe to eval here; consider using json.loads if it's a JSON string


@describe_class(
    "This agent is suitable for handling general and simple queries, such as given a topic and then writing a report, survey, or article."
)
class DeepSearch(RAGAgent):
    """
    Deep Search agent implementation for comprehensive information retrieval.

    This agent performs a thorough search through the knowledge base, analyzing
    multiple aspects of the query to provide comprehensive and detailed answers.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        personalized_info_address: str = '/Users/lixiaopeng/PycharmProjects/DeepSearch/examples/personalized_summary.json',
        max_iter: int = 3,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        self.collection_router = CollectionRouter(
            llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        )
        self.text_window_splitter = text_window_splitter
        self.personalized_info_address = personalized_info_address
        self.personalized_info = None

    def _load_personalized_info(self):
        """Load personalized information from a file."""
        with open(self.personalized_info_address, 'r', encoding='utf-8') as file:
            self.personalized_info = file.read()

    def _generate_sub_queries(self, original_query: str) -> Tuple[Any, str, int]:
        self._load_personalized_info()
        chat_response = self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": create_sub_query_prompt(original_query)}
            ]
        )
        response_content = chat_response.content
        think_content, output_content = extract_think_output(response_content)
        parsed_response = parse_response(output_content)
        return parsed_response, think_content, chat_response.total_tokens

    def _perform_search(self, query: str, sub_queries: List[str]) -> Tuple[List[str], int]:
        consume_tokens = 0
        log.color_print(f"<think> Trying to solve problem via local knowledge </think>\n")
        selected_collections, n_token_route = self._route_search(query)
        consume_tokens += n_token_route
        all_retrieved_results = self._local_search(query, sub_queries, selected_collections)
        evaluation_result, evaluate_tokens = self._evaluate_local_results(query, all_retrieved_results)
        consume_tokens += evaluate_tokens

        if evaluation_result == "YES":
            return all_retrieved_results, consume_tokens
        else:
            return self._perform_web_search(query, sub_queries, consume_tokens, all_retrieved_results)

    def _route_search(self, query: str) -> Tuple[List[str], int]:
        if self.route_collection:
            return self.collection_router.invoke(query=query, dim=self.embedding_model.dimension)
        else:
            return self.collection_router.all_collections, 0

    def _local_search(self, query: str, sub_queries: List[str], selected_collections: List[str]) -> List[str]:
        all_retrieved_results = []
        query_vector = self.embedding_model.embed_query(query)
        for collection in selected_collections:
            log.color_print(f"<local search> Searching [{query}] in [{collection}]...  </local search>\n")
            retrieved_results = self.vector_db.search_data(collection=collection, vector=query_vector, top_k=2)
            if not retrieved_results:
                continue
            for retrieved_result in retrieved_results:
                result_is_accepted = self._evaluate_retrieved_chunk(query, sub_queries, retrieved_result.text)
                if result_is_accepted:
                    all_retrieved_results.append(retrieved_result.text)
        return all_retrieved_results

    def _evaluate_retrieved_chunk(self, query: str, sub_queries: List[str], retrieved_chunk: str) -> bool:
        chat_response = self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": RERANK_PROMPT.format(query=[query] + sub_queries, retrieved_chunk=f"<chunk>{retrieved_chunk}</chunk>")}
            ]
        )
        response_content = chat_response.content.strip()
        think_content, output_content = extract_think_output(response_content)
        log.color_print(f"<think> {think_content} </think>\n")
        return "YES" in output_content and "NO" not in output_content

    def _evaluate_local_results(self, query: str, all_retrieved_results: List[str]) -> Tuple[str, int]:
        evaluation_prompt = SEARCH_EVALUATION_PROMPT.format(query=query, chunks=all_retrieved_results)
        chat_response = self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        response_content = chat_response.content.strip()
        think_content, output_content = extract_think_output(response_content)
        log.color_print(f"<think> {think_content} </think>\n")
        revised_query = re.search(r'<revised query>(.*?)</revised query>', response_content, re.DOTALL)
        return "YES" in output_content and "NO" not in output_content, chat_response.total_tokens

    def _perform_web_search(self, query: str, sub_queries: List[str], consume_tokens: int, all_retrieved_results: List[str]) -> Tuple[List[str], int]:
        api = create_search_api(search_provider="serper", serper_api_key="ee8aee330b9c12780b9880606b7b37bf6f946e6a")
        results = api.get_sources(query, num_results=2)
        web_search_results = [result['snippet'] for result in results.data['organic']]

        accepted_web_results = []
        for web_result in web_search_results:
            result_is_accepted = self._evaluate_retrieved_chunk(query, sub_queries, web_result)
            if result_is_accepted:
                accepted_web_results.append(web_result)

        all_retrieved_results.extend(accepted_web_results)
        return all_retrieved_results, consume_tokens

    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant documents from the knowledge base for the given query.
        """
        return self.async_retrieve(original_query, **kwargs)

    # Further methods such as async_retrieve, generate_gap_queries, etc., can be similarly modularized
