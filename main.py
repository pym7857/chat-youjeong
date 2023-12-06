__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

load_dotenv()

# 제목
st.title("Chat-YouJeong 📮")
st.write('---')

# Documnet Loaders
loader = TextLoader("./seed.md")
pages = loader.load_and_split()  # 페이지 별로 쪼갬

# Document Transformers
text_splitter: list = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,  # N글자 단위로 자르겠다.
    chunk_overlap  = 20,  # 겹치는 글자
    length_function = len,
    is_separator_regex = False,
)
texts: list = text_splitter.split_documents(pages)
# print(texts[0])

# Text embedding models(임베딩, OpenAI API, 유료)
# 정보를 가져오는 능력 = 임베딩
# 임베딩 모델 쓰려면, tiktoken 필요 
embeddings_model = OpenAIEmbeddings()  # .env에 있는 KEY를 자동으로 읽어옴

# load it into Chroma (now, in-memory)
vectordb = Chroma.from_documents(texts, embeddings_model)

# Question
st.header("YouJeong GPT에게 질문해보세요!!")
question = st.text_input("질문을 입력하세요.")

if st.button("질문하기"):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    result = qa_chain({"query": question})
    st.write(result['result'])
