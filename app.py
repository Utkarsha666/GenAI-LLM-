from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from PIL import Image
import PyPDF2 as pdf
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEnbeddings(model="models/embedding-001")
    vector_store=FAISS.from_text(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

## Streamlit app

st.set_page_config(page_title="Resume Expert")
st.header("ATS Tracking System")
job_description=st.text_area("Job Description ", key="input")
uploaded_file=st.file_uploader("Upload Resume",type="pdf",help="Please uplaod the pdf")

if uploaded_file is not None:
    resume_text = input_pdf_text(uploaded_file)
    st.write("PDF Uploaded")
    submit1 = st.button("Tell me about the resume")
    submit2 = st.button("Percentage Match")

    input_prompt = f"""
        You are an strict strict skilled ATS (Application Tracking System) with a deep understanding of Data Science, AI, Machine Learning, Big Data, Data Engineering, DevOps or Full Stack Web Development
        Your task is to review the provided resume {resume_text} againt the job description {job_description} for these profiles,
        Please share your professional evaluation on wheather the candidate's total experience in years aligns with job requirement, it should be strict and accurate.
        Note to be strict while evaluating the resume.
            - Experience ( in years) 
            - Skills
            - Summary ( of Projects)
            - typo errors
            
        Your Conclusion about the candidate. ( Without Table 50 words)
        do not make up the additional skills, only compare skills against job description \n\n
        the response should be in seperates table for every row easier readibility

"""

    input_prompt2 = f"""
        You are an strict skilled ATS (Application Tracking System) with a deep understanding of Data Science, AI, Full Stack Web Development,
        Big Data, Data Engineering, DevOps, Data Analyst and deep ATS functionality,
        your task is to evaluate the resume: {resume_text} against provided job description: {job_description}.
        Please share your professional evaluation on wheather the candidate's experience in years aligns with job requirement, it should be strict and accurate.
        Note to be strict while evaluating the result.

        The response should include:
            - Experience in Years.
            - Skills Matched
            - Percentage matched
            - A strict evaluation on whether the candidate should be hired or not hire ( Yes / No )
        
        Reasoning Hire or Not Hire ( Without Table 50 words )
            
        do not make up the additional skills, only compare it with skills against job description \n\n
        the response should be in seperates table for every row easier readibility
"""

    if submit1:
        response = get_gemini_response(input_prompt)
        st.subheader("Report")
        st.write(response)

    elif submit2:
        response = get_gemini_response(input_prompt2)
        st.subheader("Report")
        st.write(response)
else:
    st.write("Please Upload a Resume.pdf")