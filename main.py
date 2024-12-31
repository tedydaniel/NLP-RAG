from dotenv import load_dotenv, find_dotenv
import os
import torch

from transformers import pipeline

# from langchain_community.llms import OpenAI # not free anymore
# from langchain_nvidia_ai_endpoints import ChatNVIDIA # requires API
# load_dotenv(find_dotenv())

# print(os.environ.get("NVIDIA_API_KEY"))
# llm = ChatNVIDIA(model_name="mistralai/mixtral-8x7b-instruct-v0.1")
# llm = ChatNVIDIA(model="ai-llama2-70b", max_tokens=1000)
# result = llm.invoke("What interfaces does Triton support?")
# print(result.content)


pipe = pipeline("text-generation", "microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16)
response = pipe("Hi, tell me about planes", max_new_tokens=24)
print(response)