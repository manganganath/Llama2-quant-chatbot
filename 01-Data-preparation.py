# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Data preparation
# MAGIC
# MAGIC To be able to specialize our mode, we need a list of Q&A that we'll use as training dataset.
# MAGIC
# MAGIC For this demo, we'll specialize our model using Stack Exchange dataset. 
# MAGIC
# MAGIC Let's start with a simple data pipeline ingesting the Stack Exchange dataset, running some cleanup & saving it for further training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Downloading and extracting the raw dataset
# MAGIC
# MAGIC We'll focus on quant question, and download the quant dataset
# MAGIC
# MAGIC - Grab the quant StackExchange dataset
# MAGIC - Un-7zip it (needs `7z` installed)
# MAGIC - Copy out the `Posts.xml`
# MAGIC - Parse it with `spark-xml`
# MAGIC
# MAGIC *Note that for a real-world scenario, we would be retrieving our data from external systems such as message queue (kafka), SQL database, blob storage...*

# COMMAND ----------

# DBTITLE 1,Extract the dataset using sh command
# MAGIC %sh
# MAGIC #To keep it simple, we'll download and extract the dataset using standard bash commands 
# MAGIC #Install 7zip to extract the file
# MAGIC apt-get install -y p7zip-full
# MAGIC
# MAGIC rm -rf /tmp/quant || true
# MAGIC mkdir -p /tmp/quant
# MAGIC cd /tmp/quant
# MAGIC #Download & extract the quant archive
# MAGIC curl -L https://archive.org/download/stackexchange/quant.stackexchange.com.7z -o quant.7z
# MAGIC 7z x quant.7z 
# MAGIC #Move the dataset to our main bucket
# MAGIC rm -rf /dbfs/dbdemos/product/llm/quant/raw || true
# MAGIC mkdir -p /dbfs/dbdemos/product/llm/quant/raw
# MAGIC cp -f Posts.xml /dbfs/dbdemos/product/llm/quant/raw

# COMMAND ----------

# DBTITLE 1,Our Q&A dataset is ready
# MAGIC %fs ls /dbdemos/product/llm/quant/raw

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2/ Clean & prepare our quant questions and best answers 
# MAGIC
# MAGIC Let's ingest the data using [spark xml](https://github.com/databricks/spark-xml). Make sure the library is added to your cluster configuration page as a Maven library:
# MAGIC
# MAGIC Maven coordinates: `com.databricks:spark-xml_2.12:0.16.0` (we loaded it to the cluster created by dbdemos)
# MAGIC
# MAGIC We will perform some light preprocessing on the results:
# MAGIC - Keep only questions/answers with a reasonable score
# MAGIC - Parse HTML into plain text
# MAGIC - Join questions and answers to form question-answer pairs
# MAGIC
# MAGIC *Note that this pipeline is basic. For more advanced ingestion example with Databricks lakehouse, try Delta Live Table: `dbdemos.instal('dlt_loan')`*

# COMMAND ----------

# DBTITLE 1,Review our raw Q&A dataset
quant_raw_path = "/dbdemos/product/llm/quant/raw"
print(f"loading raw xml dataset under {quant_raw_path}")
raw_quant = spark.read.format("xml").option("rowTag", "row").load(f"{quant_raw_path}/Posts.xml")
display(raw_quant)

# COMMAND ----------

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, udf, length, pandas_udf

#UDF to transform html content as text
@pandas_udf("string")
def html_to_text(html):
  return html.apply(lambda x: BeautifulSoup(x).get_text())

quant_df =(raw_quant
                  .filter("_Score >= 5") # keep only good answer/question
                  .filter(length("_Body") <= 1000) #remove too long questions
                  .withColumn("body", html_to_text("_Body")) #Convert html to text
                  .withColumnsRenamed({"_Id": "id", "_ParentId": "parent_id"})
                  .select("id", "body", "parent_id"))

# Save 'raw' content for later loading of questions
quant_df.write.mode("overwrite").saveAsTable(f"nuwan.quant.quant_dataset")
display(spark.table("nuwan.quant.quant_dataset"))

# COMMAND ----------

# DBTITLE 1,Assemble questions and answers
import pyspark.sql.functions as F

quant_df = spark.table("nuwan.quant.quant_dataset")

# Self-join to assemble questions and answers
qa_df = quant_df.alias("a").filter("parent_id IS NULL") \
          .join(quant_df.alias("b"), on=[col("a.id") == col("b.parent_id")]) \
          .select("b.id", "a.body", "b.body") \
          .toDF("answer_id", "question", "answer")
          
# Prepare the training dataset: question following with the best answers.
docs_df = qa_df.select(col("answer_id"), F.concat(col("question"), F.lit("\n\n"), col("answer"))).toDF("source", "text")
display(docs_df)

# COMMAND ----------

# DBTITLE 1,Write processed to a Delta table
docs_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"nuwan.quant.quant_training_dataset")
display(spark.table("nuwan.quant.quant_training_dataset"))

# COMMAND ----------

# Make sure you restart the python kernel to free our gpu memory if you're using multiple notebooks0
# (load the model only once in 1 single notebook to avoid OOM)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## That's it, our Q&A dataset is ready.
# MAGIC
# MAGIC In this notebook, we leverage Databricks to prepare our Q&A dataset. We're now ready to leverage Databricks Vector Search for preparing our embeddings for this dataset and doing similarity search.
# MAGIC Open the next notebook [02-Vector-indexing]($./02-Vector-indexing).

# COMMAND ----------


