from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# STEP 1


result=YouTubeTranscriptApi().fetch('ad79nYk2keg', languages=['en'])

transcript = " ".join(chunk.text for chunk in result.snippets)


# STEP 2

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

text_chunks=text_splitter.create_documents([transcript])


# STEP 3 
embeddings=OpenAIEmbeddings()
vector_store=FAISS.from_documents(documents=text_chunks,embedding=embeddings)



# STEP 4 

retriever= vector_store.as_retriever(search_type='mmr',search_kwargs={"k":2})


user_question="how AI works"

# STEP 5

context_docs=retriever.invoke(user_question)

context= "".join([doc.page_content for doc in context_docs])

# print(context)
# for index,doc in enumerate(context_docs):
#     print(f"Document : {index} \n")
#     print(doc.page_content)
#     print("\n")

# STEP 6 PREPARE PROMPT
prompt=PromptTemplate(template="""You are a helpful assistant. Answer ONLY from the provided transcript context.If the context is insufficient, just say you don't know.
                                {context}
                                Question: {question}""",
                                input_variables=['context','question'])

llm=ChatOpenAI(model='gpt-4o')

parser=StrOutputParser()

prompt=prompt.invoke({'context':context,'question':user_question})
result=llm.invoke(prompt)



print(parser.invoke(result.content))
