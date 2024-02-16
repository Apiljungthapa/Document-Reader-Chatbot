import streamlit as st
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


def main():
    st.set_page_config(page_title="Contextual Query Assistant", layout="wide")

    st.title("Contextual Query Assistant")

    st.sidebar.title("Options")
    uploaded_files = st.sidebar.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        pdf_texts = []
        for uploaded_file in uploaded_files:
            text = ""
            with uploaded_file as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
            pdf_texts.append(text)
        context = "\n".join(pdf_texts)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(context)

        google_api_key = "AIzaSyComu6mjTazMZkqoUzu0MiMbd7ys4DMzhY"

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

        prompt_template = PromptTemplate(template="""
        Offer a detailed response to the question using the given information. 
        Ensure to include all pertinent details. If the answer cannot be found 
        in the provided information, indicate "answer is not available in the context" 
        without providing incorrect information.

        Given Context:
        {context}?

        Question:
        {question}

        Answer:
        """, input_variables=["context", "question"])

        user_input = st.text_input("Hello User Ask me i am your Assistant. If you want a call type call me")
        if user_input.lower() == "call me":
            st.write("Please provide your contact information:")
            user_name = st.text_input("Name")
            user_phone = st.text_input("Phone Number")
            user_email = st.text_input("Email")
            
            if st.button("Submit"):
                if user_name and user_phone and user_email:
                    st.write(f"Calling {user_name} at {user_phone}...")
        else:
            question = user_input
            if st.button("Get Answer"):
                docs = vector_index.get_relevant_documents(question)

                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

                formatted_response = f"The answer is: {response['output_text']}"

                st.write("Formatted Answer:", formatted_response)

if __name__ == "__main__":
    main()