# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chat Bot with langchain and Llama2
# MAGIC
# MAGIC ## Prompt engineering
# MAGIC
# MAGIC Prompt engineering is a technique used to wrap the given user question with more information to better guide the model in its anwser. Prompt engineering would typically involve:
# MAGIC - Guidance on how to answer given the usage (*ex: You are a quant and your job is to help providing the best answer*)
# MAGIC - Extra context to help your model. For example similar text close to the user question (*ex: Knowing that [Content from your internal Q&A], please answer...*)
# MAGIC - Specific instruction in the answer (*ex: Answer in Italian*) 
# MAGIC - Information on the previous questions to keep a context if you're building a chat bot (compressed as embedding)
# MAGIC
# MAGIC ### Keeping memory between multiple questions
# MAGIC
# MAGIC First of all this is expensive, but more importantly this won't support long discussion as we'll endup with a text longer than our max window size for our mdoel.
# MAGIC
# MAGIC The trick is to use a summarize model and add an intermediate step which will take the summary of our discussion and inject it in our prompt.
# MAGIC
# MAGIC We will use an intermediate summarization task to do that, using `ConversationSummaryMemory` from `langchain`.
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U \
# MAGIC   chromadb==0.3.22 \
# MAGIC   langchain==0.0.260 \
# MAGIC   xformers==0.0.20 \
# MAGIC   transformers==4.31.0 \
# MAGIC   accelerate==0.21.0 \
# MAGIC   mlflow==2.6.0 \
# MAGIC   bitsandbytes==0.41.0 \
# MAGIC   huggingface_hub \
# MAGIC   databricks-vectorsearch-preview 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=nuwan $db=quant

# COMMAND ----------

# MAGIC %md
# MAGIC ### Login to Huggingface Using Token

# COMMAND ----------

from huggingface_hub import notebook_login
notebook_login()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1/ Download our 2 embeddings model from hugging face (same as data preparation)

# COMMAND ----------

# DBTITLE 0,Create our vector database connection for context
from langchain.embeddings import HuggingFaceEmbeddings
from databricks.vector_search.client import VectorSearchClient
vs_client = VectorSearchClient()

index_name="vs_catalog.demo.quant_qna_index"

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2/ Prompt engineering with `langchain` and memory
# MAGIC
# MAGIC Now we can compose with a language model and prompting strategy to make a `langchain` chain that answers questions with a memory.

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.docstore.document import Document

def build_qa_chain():
  torch.cuda.empty_cache()
  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  You are a quant and your job is to help providing the best answer. 
  You may use information in the following paragraphs to answer the question at the end. If you don't know, say that you do not know.
  
  {context}

  {chat_history}

  Question: {human_input}

  Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'human_input', 'chat_history'], template=template)

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  model_id = "meta-llama/Llama-2-7b-chat-hf"

  model_config = AutoConfig.from_pretrained(model_id)

  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
  )

  tokenizer = AutoTokenizer.from_pretrained(model_id)

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  instruct_pipeline = pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
  )
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

  # Add a summarizer to our memory conversation
  # Let's make sure we don't summarize the discussion too much to avoid losing to much of the content

  # Models we'll use to summarize our chat history
  # We could use one of these models: https://huggingface.co/models?filter=summarization. facebook/bart-large-cnn gives great results, we'll use t5-small for memory
  summarize_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
  summarize_tokenizer = AutoTokenizer.from_pretrained("t5-small", padding_side="left", model_max_length = 512)
  pipe_summary = pipeline("summarization", model=summarize_model, tokenizer=summarize_tokenizer) #, max_new_tokens=500, min_new_tokens=300
  # langchain pipeline doesn't support summarization yet, we added it as temp fix in the companion notebook _resources/00-init 
  hf_summary = HuggingFacePipeline_WithSummarization(pipeline=pipe_summary)
  #will keep 500 token and then ask for a summary. Removes prefix as our model isn't trained on specific chat prefix and can get confused.
  memory = ConversationSummaryBufferMemory(llm=hf_summary, memory_key="chat_history", input_key="human_input", max_token_limit=500, human_prefix = "", ai_prefix = "")

  # Set verbose=True to see the full prompt:
  print("loading chain, this can take some time...")
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True, memory=memory)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the Chain for Simple Question Answering
# MAGIC
# MAGIC That's it! it's ready to go. Define a function to answer a question and pretty-print the answer, with sources:

# COMMAND ----------

class ChatBot():
  def __init__(self):
    self.reset_context()

  def reset_context(self):
    self.sources = []
    self.discussion = []
    # Building the chain will load Llama2 and can take some time depending on the model size and your GPU
    self.qa_chain = build_qa_chain()
    displayHTML("<h1>Hi! I'm your quant bot. How Can I help you today?</h1>")

  def get_similar_docs(self, question, similar_doc_count):
    docs = list()
    result = vs_client.similarity_search(
        index_name = index_name,
        query_text = question,
        columns = ["source", "text"],
        num_results = similar_doc_count)
    
    for i in range(0, similar_doc_count):
      docs.append(Document(page_content=result['result']['data_array'][i][1][1:-3], metadata={"source": int(result['result']['data_array'][i][0][1:-1])}))

    return docs

  def chat(self, question):
    # Keep the last 3 discussion to search similar content
    self.discussion.append(question)
    similar_docs = self.get_similar_docs(" \n".join(self.discussion[-3:]), similar_doc_count=2)
    # Remove similar doc if they're already in the last questions (as it's already in the history)
    similar_docs = [doc for doc in similar_docs if doc.metadata['source'] not in self.sources[-3:]]

    result = self.qa_chain({"input_documents": similar_docs, "human_input": question})
    # Cleanup the answer for better display:
    answer = result['output_text'].capitalize()
    result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
    result_html += f"<p><blockquote style=\"font-size:18px\">{answer}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
      source_id = d.metadata["source"]
      self.sources.append(source_id)
      result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"https://quant.stackexchange.com/a/{source_id}\">{source_id}</a>)</blockquote></p>"
    displayHTML(result_html)

chat_bot = ChatBot()

# COMMAND ----------

# MAGIC %md 
# MAGIC Try asking a quant question!

# COMMAND ----------

chat_bot.chat("How does the Black-Scholes model work?")

# COMMAND ----------

chat_bot.chat("What are the assumptions it makes?")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our chatbot is ready!
