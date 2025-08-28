# Customer Support Agent (EURI-only, no torch)

This project uses the EURI/EURON embeddings API to index and search your docs with FAISS.

## Quick start

1. Create `.env` with your EURI key (see `.env` example).
2. Put your docs into `data/docs.jsonl` (see sample below).
3. Build the index:
   ```bash
   python -m src.pipeline.build_index --in data/docs.jsonl --out-dir data/index --batch-size 64








(ranenv) D:\ToDoAug18_Atal_Cust_Supp>python 1_createEmbeddingOnly.py
Embedding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.31s/it]

Saved:
  Embeddings -> D:\ToDoAug18_Atal_Cust_Supp\data\processed\docs.embeddings.npy
  Metadata   -> D:\ToDoAug18_Atal_Cust_Supp\data\processed\docs.metadata.jsonl
  Info       -> D:\ToDoAug18_Atal_Cust_Supp\data\processed\embedding_info.json
Shape: (12, 1536)

(ranenv) D:\ToDoAug18_Atal_Cust_Supp>python 2_create_FAISS_IndexOnly.py

Loaded embeddings: data\processed\docs.embeddings.npy  shape=(12, 1536)
FAISS index type: IndexFlatIP   ntotal=12

Done.
 Index   -> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.index
 Metadata-> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.metadata.jsonl
 Info    -> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.index_info.json

(ranenv) D:\ToDoAug18_Atal_Cust_Supp>python 2_create_FAISS_IndexOnly.py

Loaded embeddings: data\processed\docs.embeddings.npy  shape=(12, 1536)
FAISS index type: IndexFlatIP   ntotal=12

Done.
 Index   -> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.index
 Metadata-> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.metadata.jsonl
 Info    -> D:\ToDoAug18_Atal_Cust_Supp\data\index\docs.faiss.index_info.json

(ranenv) D:\ToDoAug18_Atal_Cust_Supp>python 3_search_FAISS_ManualQueriesOnly.py --query "refund policy for damaged items" --k 5

=== refund policy for damaged items ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.631   | doc-002    | Refunds for Damaged Items    | If an item arrives damaged or defective, contact support within 7 days…
2    | 0.482   | doc-001    | Return Policy                | You may return items within 30 days of delivery for a full refund if t…
3    | 0.453   | doc-010    | Warranty Information         | Most products include a 1-year limited warranty covering manufacturing…
4    | 0.372   | doc-003    | Exchange Policy              | Exchanges are available for size or color within 30 days of delivery. …
5    | 0.353   | doc-007    | Canceling Orders             | Orders can be canceled within 60 minutes of placement from your accoun…

(ranenv) D:\ToDoAug18_Atal_Cust_Supp>python 3_search_FAISS_ManualQueriesOnly.py --queries-file queries.txt --k 5 --out-csv data\eval\search_results.csv

=== refund policy for damaged items ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.631   | doc-002    | Refunds for Damaged Items    | If an item arrives damaged or defective, contact support within 7 days…
2    | 0.482   | doc-001    | Return Policy                | You may return items within 30 days of delivery for a full refund if t…
3    | 0.453   | doc-010    | Warranty Information         | Most products include a 1-year limited warranty covering manufacturing…
4    | 0.372   | doc-003    | Exchange Policy              | Exchanges are available for size or color within 30 days of delivery. …
5    | 0.353   | doc-007    | Canceling Orders             | Orders can be canceled within 60 minutes of placement from your accoun…

=== return window length ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.188   | doc-001    | Return Policy                | You may return items within 30 days of delivery for a full refund if t…
2    | 0.187   | doc-003    | Exchange Policy              | Exchanges are available for size or color within 30 days of delivery. …
3    | 0.149   | doc-005    | International Shipping & Cus | International orders typically arrive in 7–14 business days. Customs d…
4    | 0.137   | doc-011    | Support Hours & Contact Chan | Our support team is available Monday–Friday, 9 AM–6 PM local time, via…
5    | 0.136   | doc-010    | Warranty Information         | Most products include a 1-year limited warranty covering manufacturing…

=== how long is shipping ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.563   | doc-004    | Shipping Times               | Standard shipping takes 5–7 business days in the contiguous U.S. Exped…
2    | 0.504   | doc-005    | International Shipping & Cus | International orders typically arrive in 7–14 business days. Customs d…
3    | 0.395   | doc-003    | Exchange Policy              | Exchanges are available for size or color within 30 days of delivery. …
4    | 0.385   | doc-007    | Canceling Orders             | Orders can be canceled within 60 minutes of placement from your accoun…
5    | 0.337   | doc-002    | Refunds for Damaged Items    | If an item arrives damaged or defective, contact support within 7 days…

=== warranty duration and coverage ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.593   | doc-010    | Warranty Information         | Most products include a 1-year limited warranty covering manufacturing…
2    | 0.341   | doc-002    | Refunds for Damaged Items    | If an item arrives damaged or defective, contact support within 7 days…
3    | 0.279   | doc-001    | Return Policy                | You may return items within 30 days of delivery for a full refund if t…
4    | 0.267   | doc-003    | Exchange Policy              | Exchanges are available for size or color within 30 days of delivery. …
5    | 0.250   | doc-011    | Support Hours & Contact Chan | Our support team is available Monday–Friday, 9 AM–6 PM local time, via…

=== cancel my order ===
rank | score   | id         | title                        | snippet
-----------------------------------------------------------------------------------------------------------------------------------
1    | 0.593   | doc-007    | Canceling Orders             | Orders can be canceled within 60 minutes of placement from your accoun…
2    | 0.348   | doc-008    | Payment Methods & Authorizat | We accept major credit cards, PayPal, and store credit. Some banks pla…
3    | 0.314   | doc-002    | Refunds for Damaged Items    | If an item arrives damaged or defective, contact support within 7 days…
4    | 0.302   | doc-001    | Return Policy                | You may return items within 30 days of delivery for a full refund if t…
5    | 0.282   | doc-005    | International Shipping & Cus | International orders typically arrive in 7–14 business days. Customs d…

Saved CSV -> D:\ToDoAug18_Atal_Cust_Supp\data\eval\search_results.csv

(ranenv) D:\ToDoAug18_Atal_Cust_Supp>