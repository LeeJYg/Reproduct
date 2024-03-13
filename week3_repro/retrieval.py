import urllib
from urllib.request import urlopen, Request

import urllib.request
from bs4 import BeautifulSoup

import requests
import json
import sys

def google_search(query: str, api_key: str, cse_id: str, K) -> str:
    search_url = f"https://www.googleapis.com/customsearch/v1?q={urllib.parse.quote(query)}&cx={cse_id}&key={api_key}"
    response = requests.get(search_url)
    result = json.loads(response.text)
    
    ret_urls = []
    ret_titles = []
    retrieved_num = 0
    for item in result.get('items', []):
        if 'wikipedia.org' in item['link']:
            ret_urls.append(item['link'])
            ret_titles.append(item['title'])
            retrieved_num += 1
            if retrieved_num == K:
                break
    return ret_urls, ret_titles

def wiki_retrieval(search_word: str, K = 1, max_paragraph_tokens: int = 250) -> str:
  api_key = 'AIzaSyAtkCzzG-Ll-e9UXfasHJXz-lMNqhd_Oeo'
  cse_id = 'b7d71d5feec4b40a8'
  
  wikipedia_urls, wiki_titles = google_search(search_word, api_key, cse_id, K)
  
  ret_paragraphs = []
  for wikipedia_url in wikipedia_urls:
    req = urllib.request.Request(wikipedia_url)
    response = urllib.request.urlopen(req)
    soup = BeautifulSoup(response, "html.parser")

    ret = ""
    n = 0
    for p in soup.find_all('p'):
        ret += p.text[:max_paragraph_tokens]
        n += 1
        if n == 2:
            break
    ret_paragraphs.append(ret)
  
  if ret_paragraphs != []:
    return ret_paragraphs, wiki_titles
  else:
    return [], []

if __name__ == "__main__":
  wiki_retrieval("The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn")
