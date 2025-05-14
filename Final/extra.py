import sys
import pandas as pd
from PyPDF2 import PdfReader

def generate_new_id(existing_ids):
    last_id = sorted(existing_ids)[-1]
    number = int(last_id[1:]) + 1
    return f"R{number:03d}"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def add_resume_to_dataset(pdf_path, name, category, csv_path):
    df = pd.read_csv(csv_path)
    
    resume_text = extract_text_from_pdf(pdf_path)
    new_id = generate_new_id(df["Resume_ID"].tolist())
    
    new_entry = pd.DataFrame([{
        "Category": category,
        "Resume": resume_text,
        "Resume_ID": new_id,
        "Name": name
    }])
    
    updated_df = pd.concat([df, new_entry], ignore_index=True)
    updated_df.to_csv(csv_path, index=False)
    return new_id

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python extra.py <pdf_path> <name> <category> <csv_path>")
        sys.exit(1)
        
    try:
        new_id = add_resume_to_dataset(
            sys.argv[1], 
            sys.argv[2], 
            sys.argv[3], 
            sys.argv[4]
        )
        print(f"✅ Resume added successfully with ID {new_id}.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
