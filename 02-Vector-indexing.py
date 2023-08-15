# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Data Preparation with Databricks Lakehouse
# MAGIC
# MAGIC <img style="float: right" width="600px" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/llm-dolly/llm-dolly-data-prep-small.png">
# MAGIC
# MAGIC To be able to specialize our mode, we need a list of Q&A that we'll use as training dataset.
# MAGIC
# MAGIC For this demo, we'll specialize our model using Stack Exchange dataset. 
# MAGIC
# MAGIC Let's start with a simple data pipeline ingesting the Stack Exchange dataset, running some cleanup & saving it for further training.
# MAGIC
# MAGIC We will implement the following steps: <br><br>
# MAGIC
# MAGIC <style>
# MAGIC .right_box{
# MAGIC   margin: 30px; box-shadow: 10px -10px #CCC; width:650px; height:300px; background-color: #1b3139ff; box-shadow:  0 0 10px  rgba(0,0,0,0.6);
# MAGIC   border-radius:25px;font-size: 35px; float: left; padding: 20px; color: #f9f7f4; }
# MAGIC .badge {
# MAGIC   clear: left; float: left; height: 30px; width: 30px;  display: table-cell; vertical-align: middle; border-radius: 50%; background: #fcba33ff; text-align: center; color: white; margin-right: 10px; margin-left: -35px;}
# MAGIC .badge_b { 
# MAGIC   margin-left: 25px; min-height: 32px;}
# MAGIC </style>
# MAGIC
# MAGIC
# MAGIC <div style="margin-left: 20px">
# MAGIC   <div class="badge_b"><div class="badge">1</div> Download raw Q&A dataset</div>
# MAGIC   <div class="badge_b"><div class="badge">2</div> Clean & prepare our quantitative finance questions and best answers</div>
# MAGIC   <div class="badge_b"><div class="badge">3</div> Use a Sentence 2 Vect model to transform our docs in a vector</div>
# MAGIC   <div class="badge_b"><div class="badge">4</div> Index the vector in our Vector database (Chroma)</div>
# MAGIC </div>
# MAGIC <br/>
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Fdata-science%2Fllm-dolly-chatbot%2F02-Data-preparation&cid=1444828305810485&uid=4656226177354106">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster Setup
# MAGIC
# MAGIC - Run this on a cluster with Databricks Runtime 13.0 ML GPU. It should work on 12.2 ML GPU as well.
# MAGIC - To run this notebook's examples _without_ distributed Spark inference at the end, all that is needed is a single-node 'cluster' with a GPU
# MAGIC   - A10 and V100 instances should work, and this example is designed to fit the model in their working memory at some cost to quality
# MAGIC   - A100 instances work best, and perform better with minor modifications commented below
# MAGIC - To run the examples using distributed Spark inference at the end, provision a cluster of GPUs (and change the repartitioning at the end to match GPU count)
# MAGIC
# MAGIC *Note that `bitsandbytes` is not needed if running on A100s and the code is modified per comments below to not load in 8-bit.*

# COMMAND ----------

# DBTITLE 0,Install our vector database
# MAGIC %pip install -U \
# MAGIC   chromadb==0.3.22 \
# MAGIC   langchain==0.0.260 \
# MAGIC   xformers==0.0.20 \
# MAGIC   transformers==4.31.0 \
# MAGIC   accelerate==0.21.0 \
# MAGIC   mlflow==2.5.0 \
# MAGIC   bitsandbytes==0.41.0 \
# MAGIC   huggingface_hub \
# MAGIC   databricks-vectorsearch-preview 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Create a vector index
# MAGIC
# MAGIC The Python SDK provides a VectorSearchClient as the primary way to interact with the
# MAGIC Vector Search service.

# COMMAND ----------

# DBTITLE 1,Create a VectorSearchClient
from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# COMMAND ----------

# DBTITLE 1,Vector index destination name
index_name="vs_catalog.demo.quant_qna_index"

# COMMAND ----------

# MAGIC %md
# MAGIC The person creating the index should have the same permissions as the person who registered the embedding model endpoint.

# COMMAND ----------

client.create_index(
      source_table_name="nuwan.quant.quant_training_dataset",
      dest_index_name=index_name,
      primary_key="source",
      index_column="text",
      embedding_model_endpoint_name="e5-small-v2")

# COMMAND ----------

# MAGIC %md
# MAGIC The vector index will start getting populated immediately after it is created, but it will take some time to finish backfilling. The time it will take to do this depends on the amount of the data in the delta table as well as the number of dimensions in the vector.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Check the status of an index
# MAGIC
# MAGIC The index_status field in the response to get_index has the state of the index and the
# MAGIC number of indexed documents, which will get updated as the index is backfilled.

# COMMAND ----------

index = client.get_index(index_name)
display(index)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3/ Similarity search with vector indexes
# MAGIC
# MAGIC Query the vector index for similar documents.

# COMMAND ----------

docs = client.similarity_search(
        index_name = index_name,
        query_text = "How many VIX futures should I buy to hedge my portfolio?",
        columns = ["source", "text"],
        num_results = 5)

display(
  spark.createDataFrame(docs['result']['data_array'], \
  ['source','text','similarity_score']))

# COMMAND ----------

docs['result']['data_array']

# COMMAND ----------

# Make sure you restart the python kernel to free our gpu memory if you're using multiple notebooks0
# (load the model only once in 1 single notebook to avoid OOM)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## That's it, our Q&A vector index is ready.
# MAGIC
# MAGIC In this notebook, we leverage Databricks Vector Search for preparing our embeddings and doing similarity search.
# MAGIC
# MAGIC We're now ready to use this dataset to improve our prompt context and build our quant Chat Bot! 
# MAGIC Open the next notebook [03-Prompt-engineering]($./03-Prompt-engineering).

# COMMAND ----------


