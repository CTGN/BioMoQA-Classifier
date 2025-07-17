import os
from clarivate.wos_starter.client import Configuration, ApiClient
from clarivate.wos_starter.client.api import DocumentsApi

# Configure API key
cfg = Configuration(host="https://api.clarivate.com/apis/wos-starter/v1")
cfg.api_key['ClarivateApiKeyAuth'] = os.getenv("WOS_API_KEY")
print("API key :",cfg.api_key)
try:
    with ApiClient(cfg) as client:
        docs_api = DocumentsApi(client)
        resp = docs_api.documents_get(q="DO=10.1890/02-5002", db="WOS", limit=1, page=1)
        record = resp.to_dict()
        print(record)
        print(f"Journal title: {record['hits']['title']}")
except Exception as e:
    print(f"Error occurred: {e}")
