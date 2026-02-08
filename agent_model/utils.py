import json
import re
import logging

def remove_think_tag(response): 
    # 移除<think>标签及其内容
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'\s+', ' ', response).strip()# 移除多余空白字符
    return response

def extract_dict_from_text(response):
    response = remove_think_tag(response)
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        response = response[start:end+1]
    
    if not response.endswith("}"):
        if response.endswith("\""):
            response += "}"
        else:
            response += "\"}"
        
    try:
        response_dict = json.loads(response)
    except Exception as e:
        print(response)
        print(e)
    
    for key, value in response_dict.items():
        if isinstance(value, list):
            logging.debug(f"{key} (length {len(value)}):")
            [logging.debug(v) for v in value]
        elif isinstance(value, dict):
            logging.debug(f"Total value length {len(value)}")
            for k, v in value.items():
                logging.debug(f"{k}: {v}")
        else:
            logging.debug(f"{key}: {value}")
    return response_dict