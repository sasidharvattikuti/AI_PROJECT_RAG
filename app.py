from nt import environ
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import Docx2txtLoader


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')


#Documemt Loader

loader = Docx2txtLoader("PutYourDoc.docx")
docs = loader.load()

content = docs[0].page_content

#Embedding sentence Tranceformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

reuslt = embeddings.embed_query("What is the name of the person in the resume?")

print(len(reuslt))

#Document Splitter to split the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


split=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=200)
data=split.split_documents(docs)


#Vector DB Intialization, with Emebddings, data 
from langchain_community.vectorstores import Chroma
db=Chroma.from_documents(
    documents=data, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

db.similarity_search("What is the name of the person skills?")
retriever=db.as_retriever()


#LLM Initialization
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

llm=ChatGroq(model_name="openai/gpt-oss-120b")

#Check the LLM output if you want to
#result=llm.invoke("Tell me a joke")
#result.content

#Important part we need to create chains
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>


Question:
{input}


"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
document_chain
# %% [markdown]
# HERE DOCUMENT CHAIN AND RETRIEVEL CHAIN NEED TO COMBINE


from langchain_classic.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)


#result=retrieval_chain.invoke({"input": "does phython mentioned?"})
#result
#result['answer']

# %%
#result=retrieval_chain.invoke({"input": "WHose resume it is? and what are the skills"})
#result

# %%
#result=retrieval_chain.invoke({"input": "Does this resume fit for AI Lead?"})
#result

# %% [markdown]
# User question
#      ↓
# Retriever (search)
#      ↓
# Documents
#      ↓
# Document Chain (LLM + prompt)
#      ↓
# Final answer

import streamlit as st

st.title("My First App")
query = st.text_input("What you want to know about resume")
print("The input query is ", query)
if query:
    response = retrieval_chain.invoke({"input" : query})
    st.write(response["answer"])
    print(response)
    


