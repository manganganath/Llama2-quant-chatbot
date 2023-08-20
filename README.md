# Building you own quant chatbot with Meta Llama 2 model and Databricks Vector Search

## Building a quant Chatbot to answer our questions

In this demo, we'll be building a Chat Bot to based on Llama 2 model fine-tuned for chat (Llama-2-70b-chat). 

We will split this demo in 2 sections:

- 1/ **Data preparation**: ingest and clean our Q&A dataset, transforming them as embedding in a vector database.
- 2/ **Q&A inference**: leverage Llama 2 to answer our query, leveraging our Q&A as extra context for Dolly. This is also known as Prompt Engineering.


<img style="margin: auto; display: block" width="1200px" src="https://raw.githubusercontent.com/nuwan-db/Llama2-quant-chatbot/main/_images/overview.png?token=GHSAT0AAAAAACFWHPLJUZD72SOIHZUWMMOIZHB6C4Q">
