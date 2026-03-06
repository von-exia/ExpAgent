import logging
from time import time

from baidusearch.baidusearch import search

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text
from act.tools.web_search import extract_webpage_content


class BaiduSearch(Tool):
    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.key_prompt = """
[EXTRACTION GUIDELINES]
Generate precise Baidu search terms. Follow these principles:
1. Use concise and relevant search keywords
2. Include both primary subject and key aspects for complex query
3. Maximum 3 terms

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "terms": ["string"]
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""
        self.sum_prompt = """
[EXTRACTION GUIDELINES]
You have used Baidu search to obtain the relevant information. Given the retrieval results below, use them to complete the current goal. Consider:
- Provide concise explanation about the result, where did you cite from. However, keep the explanation relatively short (maximum 200 words)
- **DO NOT** conduct any calculation or assumption, just Extract the key information about the current goal

[START OF RETRIEVAL RESULTS]
{ret}
[END OF RETRIEVAL RESULTS]

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "explanation": "string"
    "result": "string"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""
        self.sum_template = """
[START OF BAIDU SEARCH RESPONSE]
You have used Baidu search tool to obtain the relevant information from reliable source, search result is:
{result}
Explanation of the result:
{explanation}
[END OF BAIDU SEARCH RESPONSE]
"""

    def search_baidu(self, query, num_results=10):
        """
        Searches Baidu based on the given query and returns search results.

        Parameters:
            query (str): The search query for Baidu.
            num_results (int): Number of results to return.

        Returns:
            list: List of dictionaries containing search result info (title, abstract, url)
        """
        try:
            results = search(query, num_results=num_results)
            if not results:
                return [{"title": None, "url": None, "abstract": None, "error": f"No results found for query: {query}"}]

            pages_data = []
            for item in results:
                pages_data.append({
                    "title": item.get('title', ''),
                    "url": item.get('url', ''),
                    "abstract": item.get('abstract', '')
                })

            return pages_data
        except Exception as e:
            return [{"title": None, "url": None, "abstract": None, "error": f"Error searching Baidu: {str(e)}"}]

    def extract_terms_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        keywords = response_dict['terms']
        return keywords

    def extract_sum_from_response(self, response):
        response_dict = extract_dict_from_text(response)
        summarized_result = response_dict['result']
        explanation = response_dict['explanation']
        return summarized_result, explanation

    def execute(self, agent, query: str, rag_generator) -> str:
        self.rag = rag_generator
        self.query = query

        key_prompt = self.key_prompt.format(query=query)
        response = agent.response(key_prompt, stream=False)
        terms = self.extract_terms_from_response(response)
        if isinstance(terms, list):
            terms = terms[0]

        # Use keywords for searching
        results = self.search_baidu(terms, num_results=10)

        # Extract content from each URL using extract_webpage_content
        content_list = []
        for idx, r in enumerate(results[:5]):  # Top 5 results
            url = r['url']
            try:
                page_content = extract_webpage_content(url)
                if page_content and len(page_content) > 0:
                    content_list.append(page_content)
            except Exception as e:
                return{
                    "success": False,
                    "response": f"Error in Baidu Search tool: {e}"
                }

        # print(content_list)
        ret = ""
        # print("Start RAG for Baidu search results")
        # st = time()
        res = self.rag.execute(self.query, content_list, k=3)
        # ed = time()
        # print(f"End of RAG: cost time {(ed - st)/60:.4f} min")
        for i, r in enumerate(res, 1):
            ret += f"\nRetrival result {i}:\n{r}\n"
        logging.debug("RAG results:\n" + ret)

        sum_prompt = self.sum_prompt.format(ret=ret, query=query)
        response = agent.response(sum_prompt, stream=False)
        result, explanation = self.extract_sum_from_response(response)
        ret_sum = self.sum_template.format(result=result, explanation=explanation)

        return {
            "success": True,
            "response": ret_sum
        }

    @classmethod
    def content(cls):
        return """
Function: When you encounter ambiguous, unknown, or potentially inaccurate information, use this tool to search for relevant information on Baidu
Method: [
    Derive terms from query,
    Search the terms on Baidu,
    Extract the required information from search result,
    Explain the result
]
Return: [
    Extracted result,
    explanation of the result
]
"""
