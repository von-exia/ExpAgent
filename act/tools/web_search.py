import requests
from bs4 import BeautifulSoup
import logging
from act.tools.tool import Tool


def extract_webpage_content(url: str, max_chars: int = 500) -> str:
    """
    提取网页主要内容并返回文本

    Args:
        url: 网页 URL
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

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除不需要的标签
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # 提取主要内容
        content = soup.get_text(separator=' ', strip=True)

        return content

    except requests.exceptions.RequestException as e:
        logging.debug(f"Request error for {url}: {e}")
        return ""
    except Exception as e:
        logging.debug(f"Error parsing {url}: {e}")
        return ""


class WebSearch(Tool):
    def __init__(self):
        self._init_prompt()

    def _init_prompt(self):
        self.prompt = """
[EXTRACTION GUIDELINES]
Extract URLs from the query. Identify any web addresses mentioned in the query.

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "urls": ["string"]
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
You have retrieved webpage content. Given the retrieval results below, provide a concise summary:
- Summarize the key information from the webpage content
- Keep the explanation concise (maximum 200 words)

[START OF RETRIEVAL RESULTS]
{ret}
[END OF RETRIEVAL RESULTS]

## OUTPUT FORMAT:
Output STRICTLY according to this JSON Schema:
{{
    "summary": "string"
}}

Return only valid JSON.

[USER]
Query:
{query}

[ASSISTANT]
/no_think
"""

    def extract_urls_from_response(self, response):
        from agent_model.utils import extract_dict_from_text
        response_dict = extract_dict_from_text(response)
        urls = response_dict.get('urls', [])
        return urls

    def extract_summary_from_response(self, response):
        from agent_model.utils import extract_dict_from_text
        response_dict = extract_dict_from_text(response)
        summary = response_dict.get('summary', '')
        return summary

    def execute(self, agent, query: str, rag_generator=None) -> dict:
        # Extract URLs from query
        url_prompt = self.prompt.format(query=query)
        response = agent.response(url_prompt, stream=False)
        urls = self.extract_urls_from_response(response)

        if not urls:
            return {
                "success": False,
                "response": "No URLs found in the query."
            }

        # Extract content from each URL
        content_list = []
        for url in urls:
            try:
                page_content = extract_webpage_content(url)
                if page_content and len(page_content) > 0:
                    content_list.append(f"URL: {url}\nContent: {page_content}")
            except Exception as e:
                content_list.append(f"URL: {url}\nError: {str(e)}")

        if not content_list:
            return {
                "success": False,
                "response": "Failed to extract content from any URL."
            }

        # Summarize the content
        ret = "\n".join(content_list)
        sum_prompt = self.sum_prompt.format(ret=ret, query=query)
        response = agent.response(sum_prompt, stream=False)
        summary = self.extract_summary_from_response(response)

        return {
            "success": True,
            "response": summary
        }

    @classmethod
    def content(cls):
        return """
Function: Extract and summarize content from webpages given URLs in the query
Method: [
    Extract URLs from query,
    Fetch and extract content from each URL,
    Summarize the extracted content
]
Return: [
    Summary of the webpage content
]
"""
