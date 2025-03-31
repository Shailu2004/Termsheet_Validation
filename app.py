import streamlit as st
import pytesseract
from PIL import Image
import docx
import pdfplumber
import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Configure Tesseract only on Windows.
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On non-Windows systems, ensure Tesseract is installed and available in PATH.

# Load environment and configure the Gemini model.
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Sidebar: About, Features, and Team Members
st.sidebar.title("🚀 About System")
about_text = """
**TermSheet Validation using AI**

In the markets post-trade areas, our teams face significant challenges managing a high volume of term sheets.  
- **Challenge:** Manual validation is time-consuming, error-prone, and resource-intensive, leading to delays and inaccuracies.  
- **Technology:** Using OCR, NLP, ML, and AI-driven data extraction to streamline validation.
- **Benefits:** Increased efficiency, improved accuracy, cost savings, and enhanced compliance.
"""
st.sidebar.info(about_text)

st.sidebar.markdown("## 🌟 Features")
st.sidebar.markdown("✅ **Automated Data Extraction** from unstructured and structured documents.")  
st.sidebar.markdown("✅ **AI-driven Validation** to calculate missing parameters such as EMI, Interest Rate, and Tenure.")  
st.sidebar.markdown("✅ **Instant PDF Report Generation** with a downloadable term sheet report.")  
st.sidebar.markdown("✅ **User-Friendly Interface** with clear visual cues and interactive feedback.")

st.sidebar.markdown("## 👥 Team Members (Code-X)")
st.sidebar.markdown("""
- Shailesh Patil 
- Chetan Bochare 
- Sahil Bhoye  
- Vardhak Kore 
- Purva Vajire
""")

# Main Page Title and Problem Statement
st.title("📄 TermSheet Validation using AI")
st.markdown("""
### Problem Statement
In the post-trade markets, our teams manage a high volume of term sheets daily. The manual process is:
- **Time-Consuming & Error-Prone**
- **Resource-Intensive**
- **Risky** in terms of non-compliance

Our solution uses OCR, NLP, ML, and AI-driven data extraction to automatically validate term sheets against predefined criteria, improving efficiency, accuracy, and compliance.
""")

# Functions to extract text
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_docx(doc):
    return "".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to call the LLM for initial extraction.
def call_llm(extracted_text):
    prompt = f"""
    Extract the following parameters from the provided text. 
    If a parameter is not present, assign its value as null.
    Return the output as a valid JSON object with keys exactly as below:
    - Loan Amount
    - Interest Rate
    - Tenure/Maturity Date (months)
    - Borrower/Counterparty Name
    - Lender/Provider Name
    - EMI Amount
    - Penalties

    Text: '''{extracted_text}'''
    """
    response = model.generate_content(prompt)
    candidate = response._result.candidates[0].content.parts[0].text
    return candidate

# Function to call the LLM for validation & calculation.
def call_llm_validation(current_json):
    prompt = f"""
    You are provided with a JSON object representing loan parameters:
    {json.dumps(current_json, indent=2)}
    
    Some of the following fields might be missing or null: 
    - Interest Rate
    - EMI Amount
    - Tenure/Maturity Date (months)
    
    Using the available values, please calculate and fill in any missing fields. For example, if EMI Amount is missing, calculate it using the Loan Amount, Interest Rate, and Tenure.
    
    **IMPORTANT:** Return only the updated JSON object with all keys (and values filled in where possible) in valid JSON format. Do not include any additional text, code, or explanation.
    """
    response = model.generate_content(prompt)
    candidate = response._result.candidates[0].content.parts[0].text
    return candidate

# Function to call the LLM for explanation and feedback.
def call_llm_explanation(updated_json):
    prompt = f"""
    You are a financial analysis assistant. Given the following JSON object representing a term sheet, please provide a detailed explanation in the following format:

    Term Sheet Explanation:
    - Loan Amount: <explanation>
    - Interest Rate: <explanation>
    - Tenure/Maturity Date (months): <explanation>
    - Borrower/Counterparty Name: <explanation>
    - Lender/Provider Name: <explanation>
    - EMI Amount: <explanation>
    - Penalties: <explanation>

    Also, provide overall feedback on the term sheet quality and any compliance or risk considerations.

    JSON Object:
    {json.dumps(updated_json, indent=2)}

    Please return only the explanation text without any additional formatting or code.
    """
    response = model.generate_content(prompt)
    candidate = response._result.candidates[0].content.parts[0].text
    return candidate

# Function to clean the response text
def clean_response(response_text):
    cleaned_text = re.sub(r"```json", "", response_text)
    cleaned_text = re.sub(r"```", "", cleaned_text)
    return cleaned_text.strip()

# Function to generate a PDF report from the updated JSON data
def generate_pdf(data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add header: Valid Termsheet
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height - 50, "Valid Termsheet")
    
    # Convert JSON to a well formatted string and remove curly brackets
    json_str = json.dumps(data, indent=4).replace("{", "").replace("}", "")
    
    # Set up text object for body text.
    c.setFont("Helvetica", 12)
    text_object = c.beginText(40, height - 100)
    
    # Wrap the JSON string into multiple lines.
    for line in json_str.splitlines():
        text_object.textLine(line)
    
    c.drawText(text_object)
    c.showPage()
    c.save()
    
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Input Section
option = st.selectbox("📂 Choose an input type:", ["Image", "Email Body", "Document (PDF/DOCX/TXT)"])
extracted_text = ""

if option == "Image":
    uploaded_image = st.file_uploader("📷 Upload an Image", type=["png", "jpg", "jpeg"], key="imageUploader")
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="📌 Uploaded Image", use_column_width=True)
        extracted_text = extract_text_from_image(image)
elif option == "Email Body":
    extracted_text = st.text_area("📧 Enter Email Body:")
elif option == "Document (PDF/DOCX/TXT)":
    uploaded_file = st.file_uploader("📄 Upload a Document", type=["pdf", "docx", "txt"], key="docUploader")
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]
        if file_type == "docx":
            doc = docx.Document(uploaded_file)
            extracted_text = extract_text_from_docx(doc)
        elif file_type == "pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            extracted_text = uploaded_file.read().decode("utf-8")

if extracted_text:
    st.subheader("📜 Extracted Text:")
    st.text_area("", extracted_text, height=300)
    st.write("🚀 Sending extracted text to **Google Gemini AI** for structured extraction...")

    # Initial extraction using LLM
    llm_response = call_llm(extracted_text)
    cleaned_response = clean_response(llm_response)
    
    try:
        structured_data = json.loads(cleaned_response)
    except Exception as e:
        st.error("❌ Error parsing the response from LLM. Response received:")
        st.code(llm_response)
        st.stop()

    st.subheader("📊 Structured Output from AI:")
    st.json(structured_data)

    # Check if key parameters are missing
    keys_to_validate = ["Interest Rate", "EMI Amount", "Tenure/Maturity Date (months)"]
    missing_values = {key: value for key, value in structured_data.items() if key in keys_to_validate and value in [None, "", "null"]}
    
    if missing_values:
        st.warning("⚠️ Some parameters are missing. Initiating validation and recalculation using AI...")
        # Wait for 2 seconds with a spinner before the second LLM call
        with st.spinner("Validating and predicting missing fields..."):
            time.sleep(2)
            validation_response = call_llm_validation(structured_data)
            cleaned_validation_response = clean_response(validation_response)
        
        try:
            updated_data = json.loads(cleaned_validation_response)
        except Exception as e:
            st.error("❌ Error parsing the validation response from LLM. Response received:")
            st.code(validation_response)
            st.stop()

        st.subheader("🔄 Updated Structured Output from AI (After Validation):")
        st.json(updated_data)
    else:
        updated_data = structured_data
        st.success("✅ All parameters were successfully extracted with values!")
    
    # Generate PDF report from the updated JSON data, removing brackets
    pdf_report = generate_pdf(updated_data)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_report,
        file_name="termsheet.pdf",
        mime="application/pdf"
    )
    
    # Call LLM for explanation and feedback and display using st.markdown
    st.write("📝 Generating explanation and feedback for the term sheet...")
    with st.spinner("Getting AI explanation..."):
        time.sleep(2)
        explanation_response = call_llm_explanation(updated_data)
        cleaned_explanation = clean_response(explanation_response)
    
    st.subheader("💡 AI Explanation and Feedback:")
    st.markdown(cleaned_explanation)
else:
    st.write("Please provide input to extract text.")
