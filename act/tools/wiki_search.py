import requests
from bs4 import BeautifulSoup
import re
import wikipedia

from act.tools.tool import Tool

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
                        # "url": self._get_wikipedia_url(title),
                        "abstract": "Please use the URL to get the full text further if needed.",
                    })

            return pages_data
        except Exception as e:
            return [{"title": None, "url": None, "abstract": None, "error": f"Error searching Wikipedia: {str(e)}"}]
    
    def execute(self, agent, query: str, rag_generator) -> str:
        self.rag = rag_generator
        self.query = query

        # Improved prompt for keyword extrtool
        extrtool_prompt = """Given the user query below, extract 1-3 most relevant and concise keywords for wikipedia searching to respond the query, generate between <key_words> 1. 2. ... </key_words>. 
 
        Query: {query}
        
        Keywords: <key_words> 1. kw1 2. kw2 </key_words>"""
        
        # Send to agent for keyword extrtool
        formatted_prompt = extrtool_prompt.format(query=query)
        response = agent.response(formatted_prompt + "\n/no_think\n<key_words> 1.", stream=False).strip()
        
        def extract_keywords(response):
            # 提取<key_words>标签内的内容
            match = re.search(r"<key_words>(.+?)</key_words>", response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # print(f"标签内容: {content}")
                keywords = re.findall(r"\d+\.\s*(.+?)(?=\s*\d+\.|$)", content)
                # print(f"提取的关键词: {keywords}")
                return keywords
            return None
        
        keywords = extract_keywords(response)
        print("Key:", keywords)
        if isinstance(keywords, list):
            keywords = keywords[0]
        # print(f"Extracted keywords: {keywords}")
        
        # Use keywords for searching
        results = self.search_wikipedia(keywords)
        res = "Search Results:\n"
        content_list = []
        for idx, r in enumerate(results[:3], 1):  # Show top k results
            url = r['url']
            try:
                page_content = extract_webpage_content(url)
                if page_content or len(page_content) > 0:
                    # print(idx, "\n", page_content)
                    content_list.append(page_content)
            except Exception as e:
                pass
                
        ret = ""
        res = self.rag.execute(self.query, content_list, k=2)
        for i, r in enumerate(res, 1):
            ret += f"\nRetrival result {i}:\n{r}\n"
            
            
        res_prompt = f"""Given the user query below, acheive the goal based on the Retrival reslts. 
Goal: {query}
Retrival results: {ret}
"""
        formatted_prompt = res_prompt.format(query=query, ret = ret)
        response = agent.response(formatted_prompt + "\n/no_think", stream=False).strip()
        
        ret = f"""
## After Wikipedia search for key words: **{keywords}**
Retrival results: 
{ret}
Agent's answer:
{response}
"""
        return ret
    
    @classmethod
    def content(cls):
        return "Extracts keywords from queries and uses Wikipedia search to find relevant information"