from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader


st.title("AI-Powered HR Assistant")

# Load environment variables
load_dotenv()

# Load environment variables
load_dotenv()

# Load and process documents (run once at startup)
loader = PyPDFLoader('https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask an HR question:"):

    # Add user message and display immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get answer from LangChain QA
    with st.spinner("working..."):
        result = qa.invoke(prompt)
        answer = result['result']

    # Add assistant message and display
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)