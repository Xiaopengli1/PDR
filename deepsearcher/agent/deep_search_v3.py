import asyncio
import re
import requests

from typing import List, Tuple, Any

from deepsearcher.agent.base import RAGAgent, describe_class
# from deepsearcher.agent.collection_router import CollectionRouter
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

You must follow the following format:

<think>  Detailed thinking process </think>
<output> A python list of str of sub-queries </output>

You must put you reasoning process in <think> </think> and put the response of python list of str format in <output> </output>
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

INPUT FORMAT:
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

RERANK_PROMPT = """Determine if the retrieved chunk is helpful for answering ANY of the query questions.
Put your think process in <think> </think> and output in <output> </output>.

INPUT FORMAT:
Query Questions: {query}
Retrieved Chunk: {retrieved_chunk}

OUTPUT FORMAT:
<think>
1. Identify key elements in query question
2. Check if chunk contains matching information (keywords, concepts, data)
3. Helpful if ANY question is partially/completely addressed
</think>

<output>
ONLY "YES" OR "NO" HERE
</output>
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
Enclose your detailed thought process within <think></think> tags and put generation within <output></output>. Be sure to follow these steps:

<think>
1. Understand the user’s main objective: Comprehensively analyze the **original query** to identify the goal, expected output, and any preferences the user has shared (e.g., style, format, length).
2. Account for previous context: Review **previous sub-queries** and identify patterns or connections with **relevant document chunks** gathered from local and web searches.
3. Extract and integrate key data: Combine findings from previous steps, ensuring that you incorporate relevant insights and data that directly address the user’s query.
4. Structure and refine your response: Organize your insights into a cohesive, well-organized summary that clearly conveys the findings while ensuring clarity and relevance to the user’s needs.
</think>

<output>
Your generation content.
</output>

Original Query: {question}  
Previous Sub-Queries: {mini_questions}  
Related Chunks: {mini_chunk_str}
"""


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
        personalized_info_address: str,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        max_iter: int = 3,
        route_collection: bool = False,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the DeepSearch agent.

        Args:
            llm: The language model to use for generating answers.
            embedding_model: The embedding model to use for query embedding.
            vector_db: The vector database to search for relevant documents.
            max_iter: The maximum number of iterations for the search process.
            route_collection: Whether to use a collection router for search.
            text_window_splitter: Whether to use text_window splitter.
            **kwargs: Additional keyword arguments for customization.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.max_iter = max_iter
        self.route_collection = route_collection
        # self.collection_router = CollectionRouter(
        #     llm=self.llm, vector_db=self.vector_db, dim=embedding_model.dimension
        # )
        self.text_window_splitter = text_window_splitter

        if personalized_info_address is not None:
            with open(personalized_info_address, 'r', encoding='utf-8') as file:
                self.personalized_info = file.read()
        else:
            self.personalized_info = None

    def _generate_sub_queries(self, original_query: str) -> tuple[Any, Any, int]:
        
        chat_response = self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": SUB_QUERY_PROMPT_v2.format(original_query=original_query)}
            ]
        )
        response_content = chat_response.content


        think_content, output_content = self.extract_think_and_output(response_content)


        parsed_response, _ = self.llm.literal_eval(output_content)

        return parsed_response, think_content, chat_response.total_tokens



    def _search_chunks_from_mix(self, query: str, sub_queries: List[str]):
        consume_tokens = 0
        log.color_print(f"<think> Try Solving problem via inner knowledge </think>\n")

        all_retrieved_results = []
        all_accepted_chunk_num = 0

        ##########################
        ###### LOCAL SEARCH ######
        ##########################
        log.color_print(f"<local search> Conduct local Search </local search>\n")
        query_vector = self.embedding_model.embed_query(query)
        retrieved_results = self.vector_db.search_data(
            collection="personalized_knowledge", vector=query_vector, top_k=5
        )

        if not retrieved_results or len(retrieved_results) == 0:
            log.color_print(
                f"<local search> No relevant document chunks found! </local search>\n"
            )
        else:
            log.color_print(
                f"<local search> Retrieve {len(retrieved_results)} relevant chunks! </local search>\n"
            )
        accepted_local_chunk_num = 0
        references = set()
        for retrieved_result in retrieved_results:
            chat_response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                    {"role": "user", "content": RERANK_PROMPT.format(
                        query=[query],
                        retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>",),
                        }
                ]
            )
            consume_tokens += chat_response.total_tokens
            response_content = chat_response.content.strip()

            print("---------------LOCAL EVALUATION--------------")
            print(response_content)
            print("---------------LOCAL EVALUATION--------------")


            think_content, output_content = self.extract_think_and_output(response_content)


            log.color_print(f"<think> {think_content} </think>\n")

            if "YES" in output_content and "NO" not in output_content:
                log.color_print(f"accepted local chunk [{retrieved_result.text}]\n")

                all_retrieved_results.append(retrieved_result.text)
                accepted_local_chunk_num += 1
                references.add(retrieved_result.reference)
            
        if accepted_local_chunk_num > 0:
            log.color_print(
                f"<local search> Accept {accepted_local_chunk_num} document chunk(s) from references: {list(references)} </local search>\n"
            )
        else:
            log.color_print(
                f"<local search> No document chunk accepted from local DB! </local search>\n"
            )
        all_accepted_chunk_num += accepted_local_chunk_num



        #####################################
        ###### LOCAL SEARCH  EVALUATION######
        #####################################

        if all_accepted_chunk_num > 0:
            log.color_print(
                f"<evaluation> Start evaluation based on current local retrieved results. </evaluation>\n"
            )
            evaluation_prompt = SEARCH_EVALUATION_PROMPT.format(
                query=query,
                chunks=all_retrieved_results
            )
            chat_response = self.llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            response_content = chat_response.content.strip()

            # 使用正则表达式提取各部分
            think_pattern = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
            output_pattern = re.search(r'<output>(.*?)</output>', response_content, re.DOTALL)
            revised_query_pattern = re.search(r'<revised query>(.*?)</revised query>', response_content, re.DOTALL)

            # 提取匹配的内容
            think_content = think_pattern.group(1).strip() if think_pattern else None
            output_content = output_pattern.group(1).strip() if output_pattern else None
            revised_query_content = revised_query_pattern.group(1).strip() if revised_query_pattern else None

            log.color_print(
                f"<think> {think_content} </think>\n"
            )


            if "YES" in output_content and "NO" not in output_content:
                log.color_print(
                    f"<evaluation> Local chunks are sufficient to answer the question.</evaluation>\n"
                )
                return all_retrieved_results, consume_tokens

            else:
                log.color_print(
                    f"<Evaluation> Local chunks are not sufficient to answer the question, continue web search. </Evaluation>\n"
                )
                query, _ = self.llm.literal_eval(revised_query_content)
                query = query[0]

        #############################
        ###### EXTERNAL SEARCH ######
        #############################

        log.color_print(f"<external search> Conduct external search. </external search>\n")
        retrieved_external_results = self.search_from_wiki(query, top_k=5)
        external_search_results = []

        accepted_external_chunk_num = 0
        references = set()
        for result in retrieved_external_results:
            contents =  result["contents"]
            did = result["id"]

            chat_response = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                    {"role": "user", "content": RERANK_PROMPT.format(
                        query=[query],
                        retrieved_chunk=f"<chunk>{contents}</chunk>",),
                        }
                ]
            )
            consume_tokens += chat_response.total_tokens
            response_content = chat_response.content.strip()

            
            think_content, output_content = self.extract_think_and_output(response_content)


            log.color_print(f"<think> {think_content} </think>\n")

            if "YES" in output_content and "NO" not in output_content:
                log.color_print(f"accepted external chunk [{contents}]\n")
                external_search_results.append(contents)
                accepted_external_chunk_num += 1
                references.add(did)
        

        if accepted_external_chunk_num > 0:
            log.color_print(
                f"<external search> Accept {accepted_external_chunk_num} document chunk(s) from references: {list(references)} </external search>\n"
            )
        else:
            log.color_print(
                f"<external search> No document chunk accepted from external search source! </external search>\n"
            )
        
        all_accepted_chunk_num += accepted_external_chunk_num
        all_retrieved_results.extend(external_search_results)

        return all_retrieved_results, consume_tokens

    def _generate_gap_queries(
        self, original_query: str, all_sub_queries: List[str], all_chunks: List[str]
    ) -> Tuple[List[str], int]:
        reflect_prompt = REFLECT_PROMPT.format(
            question=original_query,
            mini_questions=all_sub_queries,
            mini_chunk_str=all_chunks
        )
        chat_response = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": reflect_prompt}
            ]
        )
        response_content = chat_response.content

        think_content, output_content = self.extract_think_and_output(response_content)

        log.color_print(f"<think> {think_content} </think>\n")
        log.color_print(f"<next round queries> {output_content} </next round queries>\n")

        result, _ = self.llm.literal_eval(output_content)
        return result, chat_response.total_tokens

    def retrieve(self, original_query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        Retrieve relevant documents from the knowledge base for the given query.

        This method performs a deep search through the vector database to find
        the most relevant documents for answering the query.

        Args:
            original_query (str): The query to search for.
            **kwargs: Additional keyword arguments for customizing the retrieval.

        Returns:
            Tuple[List[RetrievalResult], int, dict]: A tuple containing:
                - A list of retrieved document results
                - The token usage for the retrieval operation
                - Additional information about the retrieval process
        """
        return self.async_retrieve(original_query, **kwargs)

    def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        max_iter = kwargs.pop("max_iter", self.max_iter)
        ### SUB QUERIES ###
        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0

        sub_queries, think_process, used_token = self._generate_sub_queries(original_query)
        total_tokens += used_token
        if not sub_queries:
            log.color_print("No sub queries were generated by the LLM. Exiting.")
            return [], total_tokens, {}
        else:
            log.color_print(
                f'<think> {think_process} </think>\n'
                f"<query decomposition> Break down the original query into new sub queries: {sub_queries}</query decomposition>\n"
            )
        all_sub_queries.extend(sub_queries)
        sub_gap_queries = sub_queries

        for iter in range(max_iter):
            log.color_print(f">> Solving Round: {iter + 1}\n")
            search_res_from_vectordb = []
            search_res_from_internet = []


            ################# Sequential operation #################
            search_results_all_query_one_round = []
            for index in range(len(sub_gap_queries)):
                log.color_print(
                    f">> Start solving subquery [{index+1}/{len(sub_gap_queries)}]: {sub_gap_queries[index]}\n"
                )
                result_subquery, token = self._search_chunks_from_mix(sub_gap_queries[index], sub_gap_queries)
                search_results_all_query_one_round.extend(result_subquery)
            #######################################################


            all_search_res.extend(search_results_all_query_one_round)
            log.color_print("<search> New search queries for next iteration: {sub_gap_queries}. </search>\n")

            if iter == max_iter - 1:
                log.color_print("<think> Exceeded maximum iterations. Exiting. </think>\n")
                break
            ### REFLECTION & GET GAP QUERIES ###
            log.color_print("<think> Reflecting on the search results... </think>\n")
            sub_gap_queries, consumed_token = self._generate_gap_queries(
                original_query, all_sub_queries, all_search_res
            )
            total_tokens += consumed_token
            if not sub_gap_queries or len(sub_gap_queries) == 0:
                log.color_print("<think> No new search queries were generated. Exiting. </think>\n")
                break
            else:
                log.color_print(
                    f"<think> New search queries for next iteration: {sub_gap_queries} </think>\n"
                )
                all_sub_queries.extend(sub_gap_queries)

        additional_info = {"all_sub_queries": all_sub_queries}
        return all_search_res, total_tokens, additional_info

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.

        This method retrieves relevant documents and uses the language model
        to generate a comprehensive answer to the query.

        Args:
            query (str): The query to answer.
            **kwargs: Additional keyword arguments for customizing the query process.

        Returns:
            Tuple[str, List[RetrievalResult], int]: A tuple containing:
                - The generated answer
                - A list of retrieved document results
                - The total token usage
        """
        # Query 是入口，在这里重新生成类的个性化信息
        with open(self.personalized_info_address, 'r', encoding='utf-8') as file:
            self.personalized_info = file.read()

        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        all_sub_queries = additional_info["all_sub_queries"]

        chunk_texts = all_retrieved_results

        log.color_print(
            f"<Final Generation> Summarize answer from all {len(all_retrieved_results)} retrieved chunks... </Final Generation>\n"
        )

        summary_prompt = SUMMARY_PROMPT.format(
            question=query,
            mini_questions=all_sub_queries,
            mini_chunk_str=self._format_chunk_texts(chunk_texts),
        )
        chat_response = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(personalized_info=self.personalized_info)},
                {"role": "user", "content": summary_prompt}
            ]
        )

        response_content = chat_response.content
        
        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(response_content)

        think_content, output_content = self.extract_think_and_output(response_content)




        return (
            output_content,
            all_retrieved_results,
            n_token_retrieval + chat_response.total_tokens,
        )

    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
        return chunk_str

    def extract_think_and_output(self, response_content):
        """
        从响应内容中提取 <think> 和 <output> 标签内的内容
        
        参数:
            response_content (str): 包含XML标签的原始文本内容
            
        返回:
            tuple: (think_content, output_content) 
                - think_content: <think>标签内的文本（无空白），未找到返回None
                - output_content: <output>标签内的文本（无空白），未找到返回None
        """

        think_pattern = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        output_pattern = re.search(r'<output>(.*?)</output>', response_content, re.DOTALL)

        # 提取并清理内容
        think_content = think_pattern.group(1).strip() if think_pattern else None
        output_content = output_pattern.group(1).strip() if output_pattern else None

        return think_content, output_content

    def search_from_wiki(self, query, top_k = 2):
        url = "http://127.0.0.1:8000/retrieve"

        # Example payload
        payload = {
            "queries": [query],
            "topk": top_k,
            "return_scores": True
        }

        # Send POST request
        response = requests.post(url, json=payload)

        # Raise an exception if the request failed
        response.raise_for_status()

        # Get the JSON response
        retrieved_data = response.json()

        retrieved_all = []

        retrieved_data = retrieved_data["result"][0]

        for data in retrieved_data:
            retrieved_all.append(data["document"])

        return retrieved_all


