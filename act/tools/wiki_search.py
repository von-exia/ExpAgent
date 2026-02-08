import requests
from bs4 import BeautifulSoup
import re
from time import time
import wikipedia
import logging

from act.tools.tool import Tool
from agent_model.utils import extract_dict_from_text


def extract_webpage_content(url: str, max_chars: int = 500) -> str:
    """
    提取网页主要内容并返回文本
    
    Args:
        url: 网页URL
        max_chars: 返回的最大字符数
        
    Returns:
        网页的文本内容摘要
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        # 检测编码
        if response.encoding is None:
            response.encoding = 'utf-8'
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 移除不需要的标签
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # 尝试提取主要内容
        content = ""
        content = soup.get_text(separator=' ', strip=True)

        return content
        
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing {url}: {e}") 
        return ""


class WikipediaSearch(Tool):
    def __init__(self):
        self._init_prompt()
        
    def _init_prompt(self):
        self.key_prompt = """
[EXTRACTION GUIDELINES]
Generate precise Wikipedia search terms. Follow these principles:
1. Use standard Wikipedia article titles
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
You have used wikipeida search to obtain the relevant information. Given the retrieval results below, use them to complete the current goal. Consider:
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
[START OF WIKIPEDIA SEARCH RESPONSE]
You have used wikipeida search tool to obtain the relevant information from reliable source, search result is:
{result}
Explanation of the result:
{explanation}
[END OF WIKIPEDIA SEARCH RESPONSE]
"""     
    def _get_wikipedia_url(self, query):
        """
        Get the Wikipedia URL for a given query.
        """
        query = query.replace(" ", "_") # replace spaces with underscores
        return f"https://en.wikipedia.org/wiki/{query}"
    
    def search_wikipedia(self, query, max_length=100, max_pages=10):
        """
        Searches Wikipedia based on the given query and returns multiple pages with their text and URLs.

        Parameters:
            query (str): The search query for Wikipedia.

        Returns:
            tuple: (search_results, pages_data)
                - search_results: List of search result titles
                - pages_data: List of dictionaries containing page info (title, text, url, error)
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return [{"title": None, "url": None, "abstract": None, "error": f"No results found for query: {query}"}]

            pages_data = []
            pages_to_process = search_results[:max_pages] if max_pages else search_results

            # get the pages datafsave
            
            for title in pages_to_process:
                try:
                    page = wikipedia.page(title)
                    text = page.content
                    url = page.url

                    if max_length != -1:
                        text = text[:max_length] + f"... [truncated]" if len(text) > max_length else text

                    pages_data.append({
                        "title": title,
                        "url": url,
                        "abstract": text
                    })
                except Exception as e:
                    pages_data.append({
                        "title": title,
                        "url": self._get_wikipedia_url(title),
                        "abstract": "Please use the URL to get the full text further if needed.",
                    })

            return pages_data
        except Exception as e:
            return [{"title": None, "url": None, "abstract": None, "error": f"Error searching Wikipedia: {str(e)}"}]
    
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
        results = self.search_wikipedia(terms, max_length=100, max_pages=3)
        
        # results = []
        # for term in terms:
        #     results += self.search_wikipedia(term, max_length=100, max_pages=1)
            
        content_list = []
        for idx, r in enumerate(results[:3]):  # Show top k results
            # print(idx, r)
            url = r['url']
            try:
                page_content = extract_webpage_content(url)
                if page_content or len(page_content) > 0:
                    # print(idx, "\n", page_content)
                    content_list.append(page_content)
            except Exception as e:
                return{
                    "success": False,
                    "response": f"Error in Wikipedia Search tool: {e}"
                }

                
        ret = ""
        print("Start RAG for wiki search results")
        st = time()
        res = self.rag.execute(self.query, content_list, k=3)
        ed = time()
        print(f"End of RAG: cost time {(ed - st)/60.:.4f} min")
        for i, r in enumerate(res, 1):
            ret += f"\nRetrival result {i}:\n{r}\n"
        logging.debug("RAG results:\n"+ret)
            
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
        # return "When you encounter ambiguous, unknown, or potentially inaccurate information, use this tool to search for relevant information"
        return """
Function: When you encounter ambiguous, unknown, or potentially inaccurate information, use this tool to search for relevant information
Method: [
    Derive terms from query,
    Search the terms on Wikipedia,
    Extract the required information from search result,
    Explain the result
]
Return: [
    Extracted result,
    explaination of the result   
]
"""