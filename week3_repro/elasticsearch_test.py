from typing import List, Dict
import argparse
import sys

from collections import OrderedDict
from elasticsearch import Elasticsearch

ELASTIC_PASSWORD = "Jj+zfFq6bVYkz3cFxTPW"
#CERT_FINGERPRINT = "7c75be5559f5ffabe4cf82048bc24c86e5572c22ea327aafa4f2068bc48d36b5"
CERT_FINGERPRINT = "DF:60:72:44:1D:A2:7B:D5:43:F1:15:36:50:62:B8:CE:E3:D1:26:D6:57:0E:E7:C6:FE:67:BB:96:BC:AB:0B:B8"

class ElasticsearchRetriever:
    def __init__(self, host: str = "https://34.64.217.68:9200"):
        self._es = Elasticsearch([host], 
            request_timeout=15,
            ssl_assert_fingerprint=CERT_FINGERPRINT,
            basic_auth=('elastic', ELASTIC_PASSWORD),
        )
        
        try:
            health = self._es.cluster.health()
        except Exception as e:
            print("Error accessing Elasticsearch cluster:", str(e))
            sys.exit()

    def insert_data(
        self,
        context,
        index_name: str = 'hotpotqa',
    ):
        
        if not self._es.indices.exists(index=index_name):
            self._es.indices.create(index=index_name)  # 인덱스가 없으면 생성
        else:
            self._es.indices.refresh(index=index_name)  # 인덱스가 있으면 새로고침
        
        for item in context:
            response = self._es.index(index=index_name, id=item['idx'], body=item)
            #print(f"Indexed item {item['idx']} - ID: {response['_id']}")
        
        self._es.indices.refresh(index=index_name)

        return True
    
    def retrieve_paragraphs(
        self,
        corpus_name: str = 'hotpotqa',
        query_text: str = None,
        is_abstract: bool = None,
        allowed_titles: List[str] = None,
        allowed_paragraph_types: List[str] = None,
        query_title_field_too: bool = False,
        paragraph_index: int = None,
        max_buffer_count: int = 100,
        max_hits_count: int = 10,
    ) -> List[Dict]:

        query = {
            "size": max_buffer_count,
            "_source": ["id", "title", "paragraph_text", "url", "is_abstract", "paragraph_index"],
            "query": {
                "bool": {
                    "should": [],
                    "must": [],
                }
            },
        }

        if query_text is not None:
            # must is too strict for this:
            query["query"]["bool"]["should"].append({"match": {"paragraph_text": query_text}})

        if query_title_field_too:
            query["query"]["bool"]["should"].append({"match": {"title": query_text}})

        if is_abstract is not None:
            query["query"]["bool"]["filter"] = [{"match": {"is_abstract": is_abstract}}]

        if allowed_titles is not None:
            if len(allowed_titles) == 1:
                query["query"]["bool"]["must"] += [{"match": {"title": _title}} for _title in allowed_titles]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _title}}}} for _title in allowed_titles
                ]

        if allowed_paragraph_types is not None:
            if len(allowed_paragraph_types) == 1:
                query["query"]["bool"]["must"] += [
                    {"match": {"paragraph_type": _paragraph_type}} for _paragraph_type in allowed_paragraph_types
                ]
            else:
                query["query"]["bool"]["should"] += [
                    {"bool": {"must": {"match": {"title": _paragraph_type}}}}
                    for _paragraph_type in allowed_paragraph_types
                ]

        if paragraph_index is not None:
            query["query"]["bool"]["should"].append({"match": {"paragraph_index": paragraph_index}})

        assert query["query"]["bool"]["should"] or query["query"]["bool"]["must"]

        if not query["query"]["bool"]["must"]:
            query["query"]["bool"].pop("must")

        if not query["query"]["bool"]["should"]:
            query["query"]["bool"].pop("should")

        result = self._es.search(index=corpus_name, body=query)

        retrieval = []
        if result.get("hits") is not None and result["hits"].get("hits") is not None:
            retrieval = result["hits"]["hits"]
            text2retrieval = OrderedDict()
            for item in retrieval:
                text = item["_source"]["paragraph_text"].strip().lower()
                text2retrieval[text] = item
            retrieval = list(text2retrieval.values())

        retrieval = sorted(retrieval, key=lambda e: e["_score"], reverse=True)
        retrieval = retrieval[:max_hits_count]
        for retrieval_ in retrieval:
            retrieval_["_source"]["score"] = retrieval_["_score"]
        retrieval = [e["_source"] for e in retrieval]

        if allowed_titles is not None:
            lower_allowed_titles = [e.lower().strip() for e in allowed_titles]
            retrieval = [item for item in retrieval if item["title"].lower().strip() in lower_allowed_titles]

        for retrieval_ in retrieval:
            retrieval_["corpus_name"] = corpus_name

        return retrieval

def retrieval(test_example, query):
    retriever = ElasticsearchRetriever()    
    retriever.insert_data(test_example['contexts'])
    
    #print("Retrieved Query: ", query)

    #print("\n\nRetrieving Paragraphs ...")
    results = retriever.retrieve_paragraphs(query_text=query, query_title_field_too=True, max_hits_count=2)
    
    #for result in results: print(result)
    
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="retrieve paragraphs or titles")

    args = parser.parse_args()

    retriever = ElasticsearchRetriever()

    dev_file_path = "/home/jylee/Reproduct/week3_repro/processed_data/hotpotqa/dev.jsonl"

    from prompt_generate import read_jsonl
    import random
    
    #test_example = read_jsonl(dev_file_path)[2]
    test_example = random.choice(read_jsonl(dev_file_path))
    test_example = read_jsonl(dev_file_path)[2]
    