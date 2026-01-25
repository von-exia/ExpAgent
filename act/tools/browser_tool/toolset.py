#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Browser tools
includes:
BrowserNavigateTool,
BrowserClickTool,
BrowserInputTextTool,
BrowserGetTextTool,
BrowserGetHtmlTool,
BrowserScreenshotTool,
BrowserScrollTool,
BrowserExecuteJsTool,
BrowserGetStateTool,
BrowserReadLinksTool,
BrowserNewTabTool,
BrowserCloseTabTool,
BrowserSwitchTabTool,
BrowserRefreshTool,
"""
from ..tool import Tool

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re

MAX_LENGTH = 3000  # 最大文本长度限制

def clean_html_content(html_content):
    """
    清理HTML内容，提取对LLM友好的文本
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 移除script和style标签及其内容
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()

    # 提取标题
    title = ""
    if soup.title:
        title = soup.title.string.strip() if soup.title.string else ""

    # 提取主要文本内容
    text_content = soup.get_text()

    # 清理文本：移除多余的空白字符
    lines = (line.strip() for line in text_content.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text_content = ' '.join(chunk for chunk in chunks if chunk)

    # 提取链接
    links = []
    for link in soup.find_all('a', href=True):
        link_text = link.get_text().strip()
        link_url = link['href']
        if link_text and link_url:
            links.append({"text": link_text, "url": link_url})

    # 提取图片
    images = []
    for img in soup.find_all('img'):
        img_alt = img.get('alt', '')
        img_src = img.get('src', '')
        if img_src:
            images.append({"alt": img_alt, "src": img_src})

    # 提取重要结构信息
    headings = []
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            heading_text = heading.get_text().strip()
            if heading_text:
                headings.append({"level": i, "text": heading_text})

    return {
        "title": title,
        "main_content": text_content,
        "headings": headings,
        "links": links,  
        "images": images  
    }


def structure_webpage_data(page_data):
    """
    将网页数据组织成更易理解的格式
    """
    structured_data = {
        "summary": {
            "title": page_data["title"],
            "content_length": len(page_data["main_content"]),
            "num_headings": len(page_data["headings"]),
            "num_links": len(page_data["links"]),
            "num_images": len(page_data["images"])
        },
        "content": {
            "title": page_data["title"],
            "text": page_data["main_content"]
        },
        "structure": {
            "headings": page_data["headings"],
            "key_links": page_data["links"],
            "images": page_data["images"]
        }
    }
    return structured_data

class BrowserManager:
    """全局浏览器管理器，单例模式"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.playwright = None
            self.browser = None
            self._initialized = True
            self.current_page_text = ""
    
    async def setup(self, user_data_dir: str = "./user_data", headless: bool = True):
        """初始化浏览器"""
        async with self._lock:
            if self.browser is None:
                self.playwright = await async_playwright().start()
                # self.browser = await self.playwright.chromium.launch()
                self.browser = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=headless,
                    viewport={'width': 1280, 'height': 720}
                )
    
    async def get_browser(self):
        """获取浏览器实例（确保已初始化）"""
        if self.browser is None:
            await self.setup()
        return self.browser
    
    async def close(self):
        """关闭浏览器"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.browser = None
        self.playwright = None


class BrowserProcessor(Tool):
    def __init__(self):
        self.page_cache = {}
        self.browser_manager = BrowserManager()
        
        self.toolset = {
            "browser-navigate": BrowserNavigateTool(),
            "browser-click": BrowserClickTool(),
            "browser-input_text": BrowserInputTextTool(),
        }
        self.tool_list = list(self.toolset.keys())
        self.tool_content = "\n".join([f"- {name}: {tool.content()}" for name, tool in self.toolset.items()])
        
        
    async def async_setup(self):
        """异步设置方法"""
        if self.browser_manager.browser is None:
            await self.browser_manager.setup()
        print("Browser launched.")
    
    def setup(self):
        """同步设置方法"""
        # 确保在同一个事件循环中运行
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 如果事件循环已经在运行，创建任务
            task = loop.create_task(self.async_setup())
            loop.run_until_complete(task)
        else:
            loop.run_until_complete(self.async_setup())
        
        self._loop = loop
    
    async def async_closedown(self):
        """异步关闭方法"""
        await self.browser_manager.close()
        print("Browser closed.")
    
    def closedown(self):
        """同步关闭方法"""
        if self._loop is None:
            return
        
        if self._loop.is_running():
            task = self._loop.create_task(self.async_closedown())
            self._loop.run_until_complete(task)
        else:
            self._loop.run_until_complete(self.async_closedown())

    def execute(self, agent, query):
        pass
    
    @classmethod
    def content(cls):
        return "Start a browser, which needs a URL (like https://www.example.com), to perform specific web tasks like Navigate, Click, Input Text."


class BrowserNavigateTool(Tool):
    async def _execute(self, browser, url: str) -> str:
        page = await browser.new_page()
        await page.goto(url)
        html_content = await page.text_content('body')
        cleaned_data = clean_html_content(html_content)
        structured_data = structure_webpage_data(cleaned_data)
        return structured_data, page
    
    def extract_url(self, query):
        pattern = r'(https?://[^\s]+)'
        match = re.search(pattern, query)
        if match:
            return match.group(0)
        return ""
    
    def execute(self, agent, query):
        if agent.browser_processor is None:
            raise ValueError("BrowserProcessor is not initialized in the agent.")
        
        url = self.extract_url(query)
        if not url:
            return "No valid URL found in query."
        
        print(f"Navigating to URL: {url}")
        
        # 获取浏览器管理器和事件循环
        browser_manager = agent.browser_processor.browser_manager
        loop = agent.browser_processor._loop
        
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 获取或创建页面
        async def async_navigate():
            # 执行导航
            structured_data, updated_page = await self._execute(browser_manager.browser, url)
            
            # 保存页面到缓存
            agent.browser_processor.page_cache[url] = updated_page
            browser_manager.page = updated_page  # 更新管理器中的页面
            
            return structured_data
        
        # 在事件循环中运行异步任务
        if loop.is_running():
            # 如果事件循环已经在运行，创建任务
            task = loop.create_task(async_navigate())
            try:
                structured_data = loop.run_until_complete(task)
            except Exception as e:
                return f"Navigation failed: {str(e)}"
        else:
            # 如果事件循环没有运行，直接运行
            try:
                structured_data = loop.run_until_complete(async_navigate())
            except Exception as e:
                return {
                        'success': False, 
                        'result': f"Navigation failed: {str(e)}"
                        }

        
        agent.browser_processor.browser_manager.current_page_text = structured_data["content"]["text"][:MAX_LENGTH]
        return {
                'success': True, 
                'result': f"Your browser has successfully navigated to the URL: {url}\n"
                }
    
    
    @classmethod
    def content(cls):
        return "Execute browser navigation tasks based on the URL."


class BrowserInputTextTool(Tool):
    async def _execute(self, page, label: str, text: str) -> str:
        if not page:
            return "No active pages found."

        try:
            await page.get_by_label(label).fill(text)
            html_content = await page.content()
            cleaned_data = clean_html_content(html_content)
            structured_data = structure_webpage_data(cleaned_data)
            return structured_data, page
        
        except Exception as e:
            print(f"Failed to fill element '{label}' with text '{text}'. Error: {str(e)}")
            print("Trying alternative methods...page.get_by_role(\"textbox\")")
            try:
                # await page.screenshot(path="screenshot_before_input.png", full_page=False)
                await page.get_by_role("textbox").fill(text)
                # await page.screenshot(path="screenshot_agent.png", full_page=False)

                html_content = await page.text_content('body')
                cleaned_data = clean_html_content(html_content)
                structured_data = structure_webpage_data(cleaned_data)
                return structured_data, page
            except Exception as e:
                return f"Failed to fill element '{label}' with text '{text}'. Error: {str(e)}"

    def get_prompt(self, query):
        prompt = f"""
Goal:
{query}
For now, to input text into a field on a webpage, specify the role (e.g., 'textbox'), label (e.g., 'username'), or placeholder text of the input element along with the text you want to input.
You need output the role/label or the search text in <label> and <text> tags respectively.

## Examples 1
<input aria-label="Username">
<label for="password-input">Password:</label>
<input id="password-input">

Code:
await page.get_by_label("Username").fill("john")
await page.get_by_label("Password").fill("secret")

Output:
<label>Username</label>
<text>john</text>

## Examples 2
Search keyword "AI technology" in the search box with placeholder "Search here".
<input placeholder="Search here">

Code:
await page.get_by_role("textbox").fill("AI technology")

Output:
<label>textbox</label>
<text>AI technology</text>
"""
        return prompt

    def extract_label_and_text(self, response):
        # 提取响应中的标签和文本信息
        label_pattern = r'<label>(.*?)</label>'
        text_pattern = r'<text>(.*?)</text>'
        label_match = re.search(label_pattern, response)
        text_match = re.search(text_pattern, response)
        label = label_match.group(1).strip() if label_match else ""
        text = text_match.group(1).strip() if text_match else ""
        return label, text


    def execute(self, agent, query):
        if agent.browser_processor is None:
            raise ValueError("BrowserProcessor is not initialized in the agent.")

        prompt = self.get_prompt(query)
        current_page = agent.browser_processor.browser_manager.current_page_text
        current_page = f"""
\nCurrent Page Content:\n{current_page}\n/no_think
"""
        response = agent.response(prompt+current_page)
        print("BrowserInputTextTool response:", response)
        label, text = self.extract_label_and_text(response)

        if not label or not text:
            return "No valid label or text found in response."

        print(f"Filling element '{label}' with text: {text}")

        # 获取浏览器管理器和事件循环
        browser_manager = agent.browser_processor.browser_manager
        loop = agent.browser_processor._loop

        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 获取或创建页面
        async def async_input_text():
            # 执行输入文本操作
            structured_data, updated_page = await self._execute(browser_manager.page, label, text)

            # 更新管理器中的页面
            browser_manager.page = updated_page

            return structured_data

        # 在事件循环中运行异步任务
        if loop.is_running():
            # 如果事件循环已经在运行，创建任务
            task = loop.create_task(async_input_text())
            try:
                structured_data = loop.run_until_complete(task)
            except Exception as e:
                return f"Input text failed: {str(e)}"
        else:
            # 如果事件循环没有运行，直接运行
            try:
                structured_data = loop.run_until_complete(async_input_text())
            except Exception as e:
                return {
                'success': False, 
                'result': f"Input text failed: {str(e)}"
                }

        agent.browser_processor.browser_manager.current_page_text = structured_data["content"]["text"][:MAX_LENGTH]
        return {
                'success': True, 
                'result': f"Your browser has successfully filled the input field {label} with the specified text.\n"
                }

    @classmethod
    def content(cls):
        return "Fill an input field or text area with specified text by specifying its role (e.g., 'button'), label (e.g., 'username'), or placeholder text."


class BrowserClickTool(Tool):
    async def _execute(self, page, role_or_label: str) -> str:
        if not page:
            return "No active pages found."

        try:
            # 尝试按角色查找元素
            if role_or_label.startswith("role:"):
                role = role_or_label[5:]  # 去掉 "role:" 前缀
                element = page.get_by_role(role)
                await element.click(timeout=5000)
            # 尝试按标签查找元素
            elif role_or_label.startswith("label:"):
                label = role_or_label[6:]  # 去掉 "label:" 前缀
                element = page.get_by_label(label)
                await element.click(timeout=5000)
            # 默认尝试按文本内容查找按钮
            else:
                button = await page.get_by_role("button").click()
            
            html_content = await page.text_content('body')
            cleaned_data = clean_html_content(html_content)
            structured_data = structure_webpage_data(cleaned_data)
            return structured_data, page
        except Exception as e:
            return f"Failed to click element '{role_or_label}'. Error: {str(e)}", ""

    def get_prompt(self, query):
        prompt = f"""
To perform a click action on a webpage, specify the role (e.g., 'button'), label (e.g., 'submit'), or text content of the element you want to click.
Goal:
{query}

## Examples:
For example, this method will find inputs by label "Username" and "Password" in the following DOM:
<input aria-label="Username">
<label for="password-input">Password:</label>
<input id="password-input">

Code:
await page.get_by_label("Username").click()
await page.get_by_label("Password").click()
await page.get_by_role("button").click()
await page.get_by_text("Item").click()

## Output Format:
<role_or_label>Username</role_or_label>
"""
        return prompt

    def extract_button_info(self, response):
        # 提取响应中的按钮信息
        pattern = r'<role_or_label>(.*?)</role_or_label>'
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        return ""

    def execute(self, agent, query):
        if agent.browser_processor is None:
            raise ValueError("BrowserProcessor is not initialized in the agent.")

        prompt = self.get_prompt(query)
        current_page = agent.browser_processor.browser_manager.current_page_text
        current_page = f"""
\nCurrent Page Content:\n{current_page}\n/no_think
"""
        response = agent.response(prompt+current_page)
        role_or_label = self.extract_button_info(response)

        if not role_or_label:
            return "No valid element identifier found in response."

        print(f"Clicking element: {role_or_label}")

        # 获取浏览器管理器和事件循环
        browser_manager = agent.browser_processor.browser_manager
        loop = agent.browser_processor._loop

        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 获取或创建页面
        async def async_click():
            # 执行点击操作
            structured_data, updated_page = await self._execute(browser_manager.page, role_or_label)

            # 更新管理器中的页面
            browser_manager.page = updated_page

            return structured_data

        # 在事件循环中运行异步任务
        if loop.is_running():
            # 如果事件循环已经在运行，创建任务
            task = loop.create_task(async_click())
            try:
                structured_data = loop.run_until_complete(task)
            except Exception as e:
                return f"Click failed: {str(e)}"
        else:
            # 如果事件循环没有运行，直接运行
            try:
                structured_data = loop.run_until_complete(async_click())
            except Exception as e:
                return {
                'success': False, 
                'result': f"Click failed: {str(e)}"
                }

        agent.browser_processor.browser_manager.current_page_text = structured_data["content"]["text"][:MAX_LENGTH]
        return {
                'success': True, 
                'result': f"Your browser has successfully clicked the specified element {role_or_label}.\n"
                }

    @classmethod
    def content(cls):
        return "Click an element on the webpage by specifying its role (e.g., 'button'), label (e.g., 'submit'), or text content.\n"
