import logging

from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

def web_search(question):
    web_context = ""
    try:
        if not os.getenv("SERPER_API_KEY"):
            raise ValueError("Brak klucza API Serper")

        search = GoogleSearch({
            "q": question,
            "location": "Poland",
            "hl": "pl",
            "gl": "pl",
            "num": 5,
            "api_key": os.getenv("SERPER_API_KEY")  # Dodaj klucz API
        })

        web_results = search.get_dict()

        if 'organic_results' in web_results:
            for result in web_results['organic_results']:
                web_context += f"**{result.get('title', '')}**\n{result.get('snippet', '')}**\n{result.get('link', '')}\n\n"

    except Exception as e:
        logging.error(f"Error occur when web searching {e}")
        web_context = "\n\n⚠️ Nie udało się pobrać aktualnych danych z internetu"
    return web_context
