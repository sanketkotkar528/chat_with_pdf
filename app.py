import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import pickle
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


# Embedding Configuration
os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["OPENAI_API_KEY"] = 'dfe8fa3f42104388a15a81eefc924c84'
os.environ["OPENAI_API_BASE"] = 'https://genaiexamples.openai.azure.com/'
os.environ["OPENAI_API_VERSION"] = '2023-03-15-preview'


# Sidebar Content

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown("""
    ## About
    This app is about LLM-powered chatbot built using:
    - [Langchain](https://www.langchain.com/)
    - [Streamlit](https://streamlit.io/)
    - [OpenAI](https://openai.com/)
    """)
    add_vertical_space(5)
    st.write("Made By Sanket Kotkar.")


def main():
    st.header("Chat with PDF...")

    load_dotenv()
    # Upload the PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            text += page.extract_text()
        
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=200,
                                                    length_function=len)
        
        chunks= text_spliter.split_text(text)

        # Embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{os.path.join('embeddings',store_name)}.pkl"):
            with open(f"{os.path.join('embeddings',store_name)}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            # st.write("Embeddings are loaded from the disk.")
        else:
            with open(f"{os.path.join('embeddings',store_name)}.pkl", "wb") as f:
                embeddings = OpenAIEmbeddings(deployment=os.environ['embeding_model'])
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                pickle.dump(vectorstore, f)
            # st.write("Embeddings computation completed.")

        # Accept user questions/query    
        query = st.text_input("Ask question about your PDF file.")
        if query:
            docs = vectorstore.similarity_search(query, k=3)

            llm = AzureChatOpenAI(
                    openai_api_key=os.environ['openai_key'],
                    openai_api_base=os.environ['openai_api_base'],
                    openai_api_version=os.environ['openai_api_version'],
                    deployment_name=os.environ['chatgpt35'],
                    temperature=0.8, verbose=True
                )

            chain = load_qa_chain(llm=llm, chain_type='stuff')
            with get_openai_callback() as cb:
                responce = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(responce)



if __name__ == '__main__':
    main()
