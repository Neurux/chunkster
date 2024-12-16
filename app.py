
# ================================= IMPORTS ==============================================

import streamlit as st
import os
from dotenv import load_dotenv
from io import StringIO
import re
from unstructured.partition.pdf import partition_pdf
import tiktoken
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()
#  openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()

# # Set OPENAI_API_KEY as an environment variable
# os.environ["OPENAI_API_KEY"] = openai_api_key

# ================================= FUNCTIONS ==============================================

def characterSplitter (documents, chunk_size= 100, chunk_overlap = 0):
    text_splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap= chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks

def recursiveSplitter (documents, chunk_size= 100, chunk_overlap = 0):
    text_splitter = RecursiveCharacterTextSplitter(
        # separators= "/n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    document_chunks = text_splitter.split_documents(documents)
    return document_chunks

def documentSpecificSplitting (documents):
    pass

def semanticChunker (documents, breakpoint_threshold_type = "Default"):
    if breakpoint_threshold_type == "Default":
        text_splitter = SemanticChunker(OpenAIEmbeddings())

    if breakpoint_threshold_type == "Interquartile":
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="interquartile")

    if breakpoint_threshold_type == "Gradient":
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="gradient")

    document_chunks = text_splitter.split_documents(documents)
    return document_chunks
    

def splitByTokens (documents, chunk_size= 100, chunk_overlap = 0):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks = text_splitter.split_documents(documents)
    
    return document_chunks


def agenticChunking (documents):
    pass

def highlightChunks(loaded_documents_str, document_chunks):
    # Colors with 30% opacity using rgba
    colors = [
        "rgba(255, 215, 0, 0.5)",    # Gold
        "rgba(152, 251, 152, 0.5)",  # Pale Green
        "rgba(135, 206, 250, 0.5)",  # Light Sky Blue
        "rgba(221, 160, 221, 0.5)",  # Plum
        "rgba(240, 128, 128, 0.5)",  # Light Coral
        "rgba(224, 255, 10, 0.5)",  # Light Cyan
        "rgba(250, 250, 210, 0.5)",  # Light Goldenrod
        "rgba(216, 191, 216, 0.5)",  # Thistle
        "rgba(255, 228, 181, 0.5)",  # Moccasin
        "rgba(176, 224, 230, 0.5)",  # Powder Blue
    ]
    
    # Initialize an array to store chunk information for each character
    char_chunks = [[] for _ in range(len(loaded_documents_str))]
    
    # First pass: Mark all positions where each chunk appears
    for chunk_idx, chunk in enumerate(document_chunks):
        chunk_text = chunk.page_content
        if not chunk_text:
            continue
            
        # Find all occurrences of this chunk
        current_pos = 0
        while current_pos < len(loaded_documents_str):
            pos = loaded_documents_str.find(chunk_text, current_pos)
            if pos == -1:
                break
                
            # Mark all positions covered by this chunk
            chunk_end = pos + len(chunk_text)
            for i in range(pos, min(chunk_end, len(loaded_documents_str))):
                char_chunks[i].append(chunk_idx)
            
            current_pos = pos + max(1, len(chunk_text))
    
    # Build the HTML output with overlapping highlights
    result = []
    i = 0
    while i < len(loaded_documents_str):
        if not char_chunks[i]:  # No chunks here
            result.append(loaded_documents_str[i])
            i += 1
            continue
            
        # Handle characters covered by chunks
        current_chunks = char_chunks[i]
        
        # Find the extent of the current chunk combination
        current_pos = i
        while (current_pos < len(loaded_documents_str) and 
               current_pos < len(char_chunks) and 
               char_chunks[current_pos] == current_chunks):
            current_pos += 1
            
        # Extract the text segment
        text_segment = loaded_documents_str[i:current_pos]
        
        # Create nested spans for overlapping chunks
        wrapped_text = text_segment
        for chunk_idx in current_chunks:
            color = colors[chunk_idx % len(colors)]
            wrapped_text = f'<span style="background-color: {color}">{wrapped_text}</span>'
            
        result.append(wrapped_text)
        i = current_pos
    
    # Replace newline characters with HTML line breaks
    return f'<div style="white-space: pre-wrap">{re.sub(r"\n", "<br>", "".join(result))}</div>'



# ================================= UI ==============================================
st.write("# Chunkster")
st.write("### Visualise chunks from your files")

with st.sidebar:



    uploaded_files = st.file_uploader("Upload a file", type=None, accept_multiple_files=True)
    # youtube_url = st.text_input("Youtube URL")
    
    chunking_method = st.selectbox("Choose chunking algorithm",
                                    ("Character Splitter", "Recursive Splitter", "Document Specific Splitting", "Semantic chunking", "Split by tokens", "Agentic chunking *"))
   
    if chunking_method == "Character Splitter" or chunking_method == "Recursive Splitter" or "Split by tokens":
        chunk_size = st.slider("Chunk size", 1, 500, 100)
        if chunk_size == 1:
            st.markdown("Chunk overlap is unavaible, <br> when Chunk size = 1.", unsafe_allow_html=True)
            chunk_overlap = 0
        if chunk_size >= 2:
            chunk_overlap = st.slider("Chunk overlap", 0, chunk_size-1, 0)


   
    if chunking_method == "Document Specific Splitting":
        st.write("As Agentic chunking is an LLM-based method, results might be different in each run.")

    if chunking_method == "Semantic chunking":
        breakpoint_threshold_type = st.selectbox ("Breakpoint Threshold Type", options=["Default", "Interquartile", "Gradient"])


    if chunking_method == "Agentic chunking *":
        st.write("As Agentic chunking is an LLM-based method, results might be different in each run.")
    with st.sidebar.expander("**Options**", expanded=False):
        st.write("### Chunking options")
        

# ================================= DATA PROCESSING ==============================================
documents = []
loaded_documents_content = []

if uploaded_files:
    st.write(f"Number of files uploaded: {len(uploaded_files)}")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Get the full file path of the uploaded file
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            # Save the uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if file_path.endswith((".pdf", ".docx", ".txt")):
                # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                loader = UnstructuredFileLoader(file_path)
                loaded_documents = loader.load()

                # Extend the main documents list with the loaded documents
                documents.extend(loaded_documents)
            if file_path.endswith((".png", ".jpg")):
                    
                # Use ImageCaptionLoader to load the image file
                # image_loader = ImageCaptionLoader(path_images=[file_path])

                # # Load image captions
                # image_documents = image_loader.load()

                # # Append the Langchain documents to the documents list
                # documents.extend(image_documents)
                pass
            # Append each document's page_content to loaded_documents_content
            loaded_documents_content.extend([doc.page_content for doc in loaded_documents])
            loaded_documents_str = "".join(loaded_documents_content)
else:
    st.write("No files uploaded. Using the temporary sample file.")
    temp_file_path = "temp.txt"
    loader = UnstructuredFileLoader(temp_file_path)
    loaded_documents = loader.load()
    documents.extend(loaded_documents)
    loaded_documents_content.extend([doc.page_content for doc in loaded_documents])
    loaded_documents_str = "".join(loaded_documents_content)

# st.markdown(loaded_documents_str, unsafe_allow_html=True)

if chunking_method == "Character Splitter":
    document_chunks = characterSplitter(documents, chunk_size, chunk_overlap)
if chunking_method == "Recursive Splitter":
    document_chunks = recursiveSplitter(documents, chunk_size, chunk_overlap)
if chunking_method == "Document Specific Splitting":
    document_chunks = recursiveSplitter(documents)
if chunking_method == "Semantic chunking":
    document_chunks = semanticChunker(documents, breakpoint_threshold_type)
if chunking_method == "Split by tokens":
    document_chunks = splitByTokens(documents, chunk_size, chunk_overlap)
if chunking_method == "Agentic chunking *":
    document_chunks = recursiveSplitter(documents)



# ================================= CONTENT ==============================================
st.markdown(document_chunks)
highlighted_text = highlightChunks(loaded_documents_str, document_chunks)
# st.markdown("stop")
st.markdown(highlighted_text,unsafe_allow_html=True)
    
    
    
    
    
    
    
    
    
    
    



        # 

        # except Exception:
        #     st.error(f"Unable to decode {file.name}. Please ensure it's a valid text file.")


    # if file.type =="application/pdf":
    #     st.write("Include:")
    #     st.multiselect("Descriptive Stats", ["lenghts of documents", "Tables"])
    #     img_desc_cb = st.checkbox("Images description")
    #     tables_cb = st.checkbox("Tables")
    #     try:
    #         pdf_txt = ""
    #         with fitz.open(stream=bytes_data, filetype="pdf") as pdf_doc:
    #             for page_num in range(pdf_doc.page_count):
    #                 page = pdf_doc[page_num]
    #                 pdf_txt += page.get_text()

    #             if chunking_method == "Recursive Splitter":
    #                 chunks = recursiveSplitter(pdf_txt)
    #             if chunking_method == "Document Specific Splitting":
    #                 chunks = semanticChunking(pdf_txt)
    #             if chunking_method == "Semantic chunking":
    #                 chunks = semanticChunking(pdf_txt)
    #             if chunking_method == "Agentic chunking *":
    #                 chunks = semanticChunking(pdf_txt)
    #             else:
    #                 st.write("Select a chunking method")

        #     highlighted_text = highlightChunks(pdf_txt, chunks)
        #     st.markdown(highlighted_text, unsafe_allow_html=True)

        # except Exception as e:
        #     st.error(f"Error processing PDF {file.name}: {str(e)}")













    # === GENERAL TIPS BELOW ====
        #bytes_data =  uploaded_files.getvalue() #streamlit uploaded files have byte-like structure, so can be used in this form
        # st.write(bytes_data)
        # stringio = StringIO(uploaded_files.getvalue().decode("utf-8")) #or can be converted to diffrent formats - here a string
        # st.write(stringio)
        # string_data = stringio.read() ## To read file as string, not sure how it's different from the last above one
        # st.write(string_data)

        
        
        
    #     file_details = {"file_name" : uploaded_file.name,
    #                 "file_type" : uploaded_file.type,
    #                 "file_size" : uploaded_file.size}
    # st.write("File details:", file_details)



# 

                # if file.type not in ["text/plain", "application/octet-stream", "application/pdf"]:
                #     st.write("Upload supported file type to see your chunks.")

# chunking_method = st.slider()

    # for file in uploaded_files:
        
    # with open(temp_file, "wb") as file:
    #     file.write(uploaded_files.getvalue())
    #     file_name = uploaded_files.name

    # loader = UnstructuredFileLoader(temp_file, strategy="fast")
    # text = loader.load()
    # st.write(text)
    #         # bytes_data = file.read()
    #         # # st.write("filename:", file.name, "filetype:", file.type)
    #         # # st.write(bytes_data)
    #         # if file.type in ["text/plain", "application/octet-stream"]:
    #         #     st.multiselect("Descriptive Stats", ["lenghts of documents"])
    #         #     # try:
    #         #     stringio = StringIO(bytes_data.decode("utf-8"))
    #         #     text = stringio.read()