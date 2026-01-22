import os 
import google.generativeai as genai 
from PDF_Extractor import text_extractor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st 
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Lets configure the models 

# LLM
gemini_key = os.getenv('TestProject1')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Configure Embdeding Model
embeding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')

# Let's Create the main page 
st.title(':orange[RAG ChatBot:] :blue[AI Assissted ChatBot using RAG]')
tips = '''
Follow the steps to use the applications:
* Upload the document in sidebar  .
* Write the Quey and start the chat .
'''
st.text(tips)

# Let's create the side bar
st.sidebar.title(':green[File:]')
st.sidebar.subheader(':red[Upload PDF file only.]')
pdf_file = st.sidebar.file_uploader('Upload here', type = ['pdf']) 
if pdf_file:
    st.sidebar.success('File Uploaded Successfully')
    
    file_text = text_extractor(pdf_file)
    
    # Step1: Chunking 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)  
    chunks = splitter.split_text(file_text)
    
    # Step2: Create the vector database (FAISS)
    vector_store = FAISS.from_texts(chunks,embeding_model)
    retirever = vector_store.as_retriever(search_kwargs={'k':3}) 
    
    def generate_content(query):
        # Step 3 : Retrieval (R)
        retrived_docs = retirever.invoke(query)
        context = '\n'.join([d.page_content for d in retrived_docs])
        
        #Step 4 Augmenting (A)
        augmented_prompt = f'''
        <Role> You are an helpful assistant uing RAG.
        <Goal> Answer the question asked by the user . Here is the question: {query}
        <Context> Here are th documents retrivered from the vetcor database to suport the answer
        which you have to generate {context}
        '''
        #Step5 : Generate (G)
        respsone = model.generate_content(augmented_prompt)
        return respsone.text 
    
    # Create the Chatbot in order to start in Conversation 
    # To intillize the chat we need to create History , if not created
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    # Display the history 
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.info(f"User: {msg['text']}") 
        else:
            st.warning(f"CHATBOT: {msg['text']}")
    # Input from the user using streamlit form
    with st.form('ChatBot Form',clear_on_submit=True):
        user_query = st.text_area('Ask Anything: ')
        send = st.form_submit_button('send')
        
    # start the convo and apend and queery in hostory 
    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'chatbot','text':generate_content(user_query)})
        st.rerun() 
        