# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Building and querying a vector index with Databricks Vector Search (Private Preview)
# MAGIC
# MAGIC This notebook shows how to create and query a vector index for the dataset that we prepared in the previous notebook. Note that Databricks Vector Search is in private preview and these APIs are subject to change based on the
# MAGIC feedback from customers and should not be used for production workloads.
# MAGIC
# MAGIC
# MAGIC Ensure you have the following set up:
# MAGIC - UC enabled on the workspace that you are planning to build a vector index on.
# MAGIC - A delta table that has a column with text that you want to index. 
# MAGIC - Model Serving endpoint with the embedding model you want to use

# COMMAND ----------

# DBTITLE 0,Install our vector database
# MAGIC %pip install -U databricks-vectorsearch-preview

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

# Delete an index
#client.delete_index(index_name=index_name)

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


