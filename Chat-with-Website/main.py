#import atexit
import os
import logging
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI #from langchain.chat_models import ChatOpenAI
from langchain.embeddings import GooglePalmEmbeddings #OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
#import openai
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import pytesseract
from langchain_core.messages import HumanMessage, AIMessage

# Ensure pytesseract is properly configured
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust this path based on your installation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

def solve_captcha(session, captcha_img_url, form_action, captcha_input_name, form_data):
    # Fetch the CAPTCHA image
    captcha_response = session.get(captcha_img_url)
    captcha_image = Image.open(BytesIO(captcha_response.content))

    # Use OCR to solve the CAPTCHA
    captcha_text = pytesseract.image_to_string(captcha_image).strip()

    # Add the CAPTCHA solution to the form data
    form_data[captcha_input_name] = captcha_text

    # Submit the CAPTCHA solution
    post_response = session.post(form_action, data=form_data)
    return post_response

def crawl_website(url):
    session = requests.Session()
    response = session.get(url)

    if response.status_code != 200:
        st.error(f"Failed to retrieve the website content. Status code: {response.status_code}")
        logger.error(f"Failed to retrieve the website content. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Check if a CAPTCHA is present
    captcha_img = soup.find('img', {'src': lambda x: x and 'captcha' in x})
    if captcha_img:
        captcha_url = captcha_img['src']
        if not captcha_url.startswith('http'):
            captcha_url = url + captcha_url

        # Locate CAPTCHA form and inputs
        captcha_form = soup.find('form', {'id': 'captcha_form'})
        if not captcha_form:
            st.error("CAPTCHA form not found on the page.")
            logger.error("CAPTCHA form not found on the page.")
            return None

        captcha_input_name = captcha_form.find('input', {'type': 'text'})['name']
        form_action = captcha_form['action']
        if not form_action.startswith('http'):
            form_action = url + form_action

        form_data = {input_tag['name']: input_tag.get('value', '') for input_tag in captcha_form.find_all('input', {'type': 'hidden'})}

        # Solve CAPTCHA
        post_response = solve_captcha(session, captcha_url, form_action, captcha_input_name, form_data)
        if post_response.status_code != 200:
            st.error(f"Failed to submit CAPTCHA. Status code: {post_response.status_code}")
            logger.error(f"Failed to submit CAPTCHA. Status code: {post_response.status_code}")
            return None

        soup = BeautifulSoup(post_response.content, 'html.parser')

    return soup.get_text()

def process_website_content(website_text, DB, key):
    # Split the loaded data
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=40)
    docs = text_splitter.split_text(website_text)

    # Create OpenAI embeddings
    google_embeddings = GooglePalmEmbeddings(model="models/embedding-001")

    # Create a Chroma vector database from the documents
    vectordb = Chroma.from_texts(texts=docs, embedding=google_embeddings, persist_directory=DB)
    
    return vectordb

#to delete the persistent db at exit 
#if the db isn't deleted, it causes problems
# def cleanup():
#     ABS_PATH = os.path.dirname(os.path.abspath(__file__))
#     DB_DIR = os.path.join(ABS_PATH, "db")
#     if os.path.exists(DB_DIR):
#         shutil.rmtree(DB_DIR)
#         logger.info(f"Deleted the persistent database directory: {DB_DIR}")



def main():
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "db")


    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')
    
    with st.sidebar:
        key = st.text_input("Enter Google Gemini API Key")
        url = st.text_input("Insert The website URL")
    
    if not key or not url:
        st.info("Enter Valid URL and API Key.")
    else:    
        prompt = st.chat_input("Ask a question (query/prompt)")
        
        os.environ['GOOGLE_API_KEY'] = key

        if prompt:
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                AIMessage(content="Hello, how can I help today?")
                ]
                
            if prompt is not None and prompt != "":
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                with st.chat_message("Human"):
                    st.write(prompt)
                        
            website_text = crawl_website(url)
            if website_text is None:
                return
            if "vectordb" not in st.session_state:
                st.session_state.vectordb = process_website_content(website_text, DB_DIR, key)


            # Create a retriever from the Chroma vector database
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})

            # Use a ChatOpenAI model
            llm = GoogleGenerativeAI(model='gemini-1.5-pro')

            # Create a RetrievalQA from the model and retriever
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Run the prompt and return the response
            response = qa(prompt)
            #st.write(response["result"])
            
            responseResult = response['result']
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                AIMessage(content="Hello, how can I help today?")
                ]
                    
                   
                        
            if responseResult is not None and responseResult != "":
                st.session_state.chat_history.append(AIMessage(content=responseResult))
                with st.chat_message("AI"):
                    st.write(responseResult)
                    

if __name__ == '__main__':
    main()
