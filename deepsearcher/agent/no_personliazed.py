from typing import List, Tuple
import re
from deepsearcher.agent.base import RAGAgent
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB, RetrievalResult, deduplicate_results

SUMMARY_PROMPT_1 = """You are a AI content analysis expert, please answer the provided query. Please put your thinking progress in <think> </think>, and final output in <output> </output>.

INPUT:
Original Query: {query}

OUTPUT:
<think> Your thinking progress </think>
<output> Your generation content. </output>
"""

SUMMARY_PROMPT_2 = """You are a AI content analysis expert, please answer the provided query and related chunks. Please put your thinking progress in <think> </think>, and final output in <output> </output>.
Enclose your detailed thought process within <think></think> tags and put generation within <output></output>. Be sure to follow these steps:

INPUT:
Original Query: {query}
Related Chunks: {mini_chunk_str}


OUTPUT:
<think> Your thinking progress </think>
<output> Your generation content. </output>
"""


class FreePersonalized_Zeroshot(RAGAgent):
    """
    Naive Retrieval-Augmented Generation agent implementation.

    This agent implements a straightforward RAG approach, retrieving relevant
    documents and generating answers without complex processing or refinement steps.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        top_k: int = 2,
        route_collection: bool = False,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the NaiveRAG agent.

        Args:
            llm: The language model to use for generating answers.
            embedding_model: The embedding model to use for query embedding.
            vector_db: The vector database to search for relevant documents.
            **kwargs: Additional keyword arguments for customization.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k
        self.route_collection = route_collection
        self.text_window_splitter = text_window_splitter


    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Query the agent and generate an answer based on retrieved documents.

        This method retrieves relevant documents and uses the language model
        to generate a simple answer to the query.

        Args:
            query (str): The query to answer.
            **kwargs: Additional keyword arguments for customizing the query process.

        Returns:
            Tuple[str, List[RetrievalResult], int]: A tuple containing:
                - The generated answer
                - A list of retrieved document results
                - The total token usage
        """

        summary_prompt = SUMMARY_PROMPT_1.format(query=query)
        char_response = self.llm.chat([{"role": "user", "content": summary_prompt}])
        
        response_content = char_response.content.strip()

        all_retrieved_results = None
        n_token_retrieval = 0


        # Use regex to extract sections
        think_pattern = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        output_pattern = re.search(r'<output>(.*?)</output>', response_content, re.DOTALL)

        # Extract matched content
        think_content = think_pattern.group(1).strip() if think_pattern else None
        output_content = output_pattern.group(1).strip() if output_pattern else None

        log.color_print(f"<think>{think_content}</think>")

        log.color_print("\n==== FINAL ANSWER====\n")
        log.color_print(output_content)

        return output_content, all_retrieved_results, n_token_retrieval


class FreePersonalized_Search(RAGAgent):
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

        self.text_window_splitter = text_window_splitter

        if personalized_info_address is not None:
            with open(personalized_info_address, 'r', encoding='utf-8') as file:
                self.personalized_info = file.read()
        else:
            self.personalized_info = None




    async def _search_chunks_from_mix(self, query: str, sub_queries: List[str]):
        consume_tokens = 0
        log.color_print(f"<think> Try Solving problem via inner knowledge </think>\n")

        all_retrieved_results = []

        #############################
        ###### EXTERNAL SEARCH ######
        #############################

        log.color_print(f"<external search> Conduct external search. </external search>\n")
        retrieved_external_results = self.search_from_wiki(query, top_k=5)
        external_search_results = []
        for result in retrieved_external_results:
            contents =  result["contents"]
            external_search_results.append(contents)
        

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
        return asyncio.run(self.async_retrieve(original_query, **kwargs))

    async def async_retrieve(
        self, original_query: str, **kwargs
    ) -> Tuple[List[RetrievalResult], int, dict]:
        max_iter = kwargs.pop("max_iter", self.max_iter)
        ### SUB QUERIES ###
        log.color_print(f"<query> {original_query} </query>\n")
        all_search_res = []
        all_sub_queries = []
        total_tokens = 0
        sub_queries = [original_query]

        for iter in range(max_iter):
            log.color_print(f">> Solving Round: {iter + 1}\n")
            ################# Prarallel operation #################
            search_results_all_query_one_round = []
            search_tasks = [
                self._search_chunks_from_mix(sub_queries[index], sub_queries)
                for index in range(len(sub_queries))
            ]
            
            # Execute all tasks in parallel and wait for results
            search_results = await asyncio.gather(*search_tasks)

            # Merge all results
            for result in search_results:
                result_subquery, token = result
                # total_tokens += token
                search_results_all_query_one_round.extend(result_subquery)

            search_results_all_query_one_round = self.deduplicate_results(search_results_all_query_one_round)
            #######################################################

            ################# Sequential operation #################
            # search_results_all_query_one_round = []
            # for index in range(len(sub_gap_queries)):
            #     log.color_print(
            #         f">> Start solving subquery [{index+1}/{len(sub_gap_queries)}]: {sub_gap_queries[index]}\n"
            #     )
            #     result_subquery, token = self._search_chunks_from_mix(sub_gap_queries[index], sub_gap_queries)
            #     search_results_all_query_one_round.extend(result_subquery)
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
        # Query is the entry point; reload personalized info here
        # with open(self.personalized_info_address, 'r', encoding='utf-8') as file:
        #     self.personalized_info = file.read()

        all_retrieved_results, n_token_retrieval, additional_info = self.retrieve(query, **kwargs)
        if not all_retrieved_results or len(all_retrieved_results) == 0:
            return f"No relevant information found for query '{query}'.", [], n_token_retrieval
        all_sub_queries = additional_info["all_sub_queries"]

        chunk_texts = all_retrieved_results

        log.color_print(
            f"<Final Generation> Summarize answer from all {len(all_retrieved_results)} retrieved chunks... </Final Generation>\n"
        )

        summary_prompt = SUMMARY_PROMPT_2.format(
            question=query,
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
        Extract content from <think> and <output> tags in the response.

        Args:
            response_content (str): Raw text containing XML tags.

        Returns:
            tuple: (think_content, output_content)
                - think_content: Text inside <think> tag (stripped), None if not found
                - output_content: Text inside <output> tag (stripped), None if not found
        """
        think_pattern = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        output_pattern = re.search(r'<output>(.*?)</output>', response_content, re.DOTALL)

        # Extract and clean content
        think_content = think_pattern.group(1).strip() if think_pattern else None
        output_content = output_pattern.group(1).strip() if output_pattern else None

        return think_content, output_content

    def search_from_wiki(self, query, top_k = 5):
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
    
    def deduplicate_results(self, results):
        """
        Remove duplicate results based on text content.

        This function removes duplicate results from a list of RetrievalResult objects
        by keeping only the first occurrence of each unique text content.

        Args:
            results: A list of RetrievalResult objects to deduplicate.

        Returns:
            A list of deduplicated RetrievalResult objects.
        """
        all_text_set = set()
        deduplicated_results = []
        for result in results:
            if result not in all_text_set:
                all_text_set.add(result)
                deduplicated_results.append(result)
        return deduplicated_results


