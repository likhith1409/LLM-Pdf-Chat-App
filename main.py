import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

def main():
    st.title("🤗💬 LLM Pdf Chat App")
    st.markdown('''
    ## About
    Welcome to the LLM Chat App! This tool allows you to interact with PDF documents using advanced AI, enabling you to ask questions and receive insightful answers. Upload a PDF, ask your questions, and get responses in real-time. Powered by Streamlit, LangChain, and OpenAI, this app makes engaging with your documents easier than ever.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    ''')

    st.header("Chat with PDF 💬")

    
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        
        progress_text.text("Reading PDF...")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        progress_bar.progress(30)

        
        progress_text.text("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        progress_bar.progress(60)

        
        progress_text.text("Creating/loading embeddings...")
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        progress_bar.progress(90)

       
        query = st.text_input("Ask questions about your PDF file:")
        
        if query:
            progress_text.text("Processing query...")
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write(response)

                
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({"query": query, "response": response})

                progress_text.text("")

            progress_bar.progress(100)

    if "history" in st.session_state:
        st.header("Chat History")
        for entry in st.session_state.history:
            st.write(f"**Question:** {entry['query']}")
            st.write(f"**Answer:** {entry['response']}")
            st.write("---")

if __name__ == '__main__':
    main()
