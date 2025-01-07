from dotenv import load_dotenv, find_dotenv
import os
import torch

from transformers import pipeline
from langchain.schema.runnable import RunnableLambda
from langchain_ai21 import ChatAI21, AI21Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from operator import itemgetter


load_dotenv(find_dotenv())

class Docs:
    def __init__(self):
        self.db = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions that would be good to incorporate into summary, but that are yet unknown (max 3)")


summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one. Make sure a reader can still understand everything."
    " Keep it short, but as dense and useful as possible! The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keep all of the information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
)


# model = ChatAI21(
#     api_key=os.environ.get("AI21_API_KEY"),
#     model="jamba-1.5-mini",
#     temperature=0
# )

embedder = AI21Embeddings(api_key=os.environ.get("AI21_API_KEY"))

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

docs = Docs()

print("Chunking Documents")
docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

## Make some custom Chunks to give big-picture details
doc_string = "Available Documents:"
doc_metadata = []
for chunks in docs_chunks:
    metadata = getattr(chunks[0], 'metadata', {})
    doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

extra_chunks = [doc_string] + doc_metadata

print("Constructing Vector Stores")
vecstores = [FAISS.from_texts(extra_chunks, embedder)]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

def aggregate_vstores(vectorstores):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

## Unintuitive optimization; merge_from seems to optimize constituent vector stores away
docstore = aggregate_vstores(vecstores)

convstore = FAISS.from_documents(docstore, embedding=embedder)
retriever = convstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in rhymes"),
    ("user", "{input}")
])

# print(model.invoke("Hi"))

model = RunnableLambda(lambda x : "Hi")

# from langchain_community.llms import OpenAI # not free anymore
# from langchain_nvidia_ai_endpoints import ChatNVIDIA # requires API
# load_dotenv(find_dotenv())

# print(os.environ.get("NVIDIA_API_KEY"))
# llm = ChatNVIDIA(model_name="mistralai/mixtral-8x7b-instruct-v0.1")
# llm = ChatNVIDIA(model="ai-llama2-70b", max_tokens=1000)
# result = llm.invoke("What interfaces does Triton support?")
# print(result.content)


# pipe = pipeline("text-generation", "microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16)
# response = pipe("Hi, tell me about planes", max_new_tokens=24)
# print(response)

retriever = itemgetter("input") | RunnableLambda(lambda x: docs.retrieve(x))
generator = RunnableLambda(lambda x: x)

rag = prompt | retriever | generator

print(rag.invoke({"input" : "Tell me about birds!"}))