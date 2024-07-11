from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from langchain_community.llms import OpenAI
import uuid
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_openai import OpenAI

app = FastAPI()

load_dotenv()

# S3 configuration
S3_BUCKET = "municonnect-bucket-24"
S3_REGION = "eu-north-1"
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")


s3 = boto3.client('s3', region_name=S3_REGION,
                  aws_access_key_id=S3_ACCESS_KEY,
                  aws_secret_access_key=S3_SECRET_KEY)

    # Initialize the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise HTTPException(status_code=500, detail="OpenAI API key is not set")

openai = OpenAI(api_key=openai_api_key, temperature=0)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3000",
    "http://your-frontend-domain.com",  # Add your frontend domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key, temperature=0)

class DatasetMetadata(BaseModel):
    name: str
    owner: str
    description: str
    tags: List[str]
    language: str
    is_private: bool

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("./uploads", unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_rows, fields, file_size = process_file_basic(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")        
   
    # # Upload files to S3
    main_file_url = upload_to_s3(file_path, unique_filename)
        
    return {
        "main_file_url": main_file_url,        
        "num_rows": num_rows,
        "fields": fields,
        "file_size": file_size
    }

class DatasetSummary(BaseModel):
    file_name: str
    domain: str

@app.post("/process_summary/")
async def process_summary(request: DatasetSummary):
    file_name = request.file_name
    domain = request.domain

    file_path = os.path.join("./uploads", file_name)
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not set")
     
    print(f"FILE ${file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save the dataframe to a temporary CSV file for processing
        temp_csv_path = file_path + "_temp.csv"
        df.to_csv(temp_csv_path, index=False)
        
        # Create an agent to interact with the dataset
        agent = create_csv_agent(openai, temp_csv_path, verbose=True, allow_dangerous_code=True)
        
        # Generate summary
        summary_result = agent.run(f"Please provide a clear and well thought out summary of this dataset. And if applicable, describe how it helps in the development in a typical city in Congo in the domain of {domain}.")
        summary = summary_result.split("Final Answer: ")[-1]
        
        # Remove the temporary CSV file
        os.remove(temp_csv_path)
        
        return {
            "summary": summary
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty file or invalid content")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


class Question(BaseModel):
    file_name: str
    question: str

@app.post("/chat")
async def answer_question_route(request: Question):
    file_name = request.file_name
    question = request.question

    file_path = f"./uploads/{file_name}"
    answer = get_answer_from_csv(question, file_path)

    return answer


class FileNameRequest(BaseModel):
    file_name: str

@app.post("/sample-questions")
async def generate_sample_questions(request: FileNameRequest):
    file_name = request.file_name
    file_path = os.path.join("./uploads", file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not set")

    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save the dataframe to a temporary CSV file for processing
        temp_csv_path = file_path + "_temp.csv"
        df.to_csv(temp_csv_path, index=False)
        
        # Create an agent to interact with the dataset
        agent = create_csv_agent(openai, temp_csv_path, verbose=True, allow_dangerous_code=True)
        
        # Generate sample questions
        questions_result = agent.run("Based on this dataset, generate 4 insightful sample questions that could be asked about the data. You can suggest the questions by basing yourself on the column titles to reduce the iterations. Return only the questions in a numbered list format.")
        
        # Extract questions from the result
        questions = questions_result.split("Final Answer:")[-1].strip().split("\n")
        questions = [q.strip().lstrip("1234. ") for q in questions if q.strip()]

        # Remove the temporary CSV file
        os.remove(temp_csv_path)
        
        return {
            "questions": questions[:4]  # Ensure we return exactly 4 questions
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty file or invalid content")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")        

def process_dataset_with_ai(file_path: str):

    # Create an agent to interact with the dataset
    try:
        agent = create_csv_agent(openai, file_path, verbose=True, allow_dangerous_code=True)
        
        # Generate summary
        summary_result = agent.run("Please provide a summary of this dataset in terms of what it contains and how it helps in a typical city in Congo.")
        summary = summary_result.split("Final Answer: ")[-1]
        print(f"Summary: {summary}")
        
        # Generate fields info
        fields_result = agent.run("List the fields, their types and their very short descriptions in json format.")
        fields = fields_result.split("Final Answer: ")[-1]
        print(f"Fields: {fields}")
        
        # Generate an action message based on dataset content
        action_prompt = (
            f"The following dataset contains information that can help improve municipal services. "
            f"Based on this dataset, what actions(max 4 and don't cut out text) can be taken to enhance municipal management?\n\n"
            f"Dataset Summary:\n{summary}\n\nAction Message:"
        )
        
        action_result = openai.generate([action_prompt])
        action_message = action_result.generations[0][0].text
        print(f"Token usage: {action_result.llm_output}")
        
        return summary, fields, action_message
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

def upload_to_s3(file_path: str, filename: str):
    try:
        print(f"FILE ${S3_ACCESS_KEY} ${S3_SECRET_KEY} ${S3_BUCKET} ${S3_REGION}")
        s3.upload_file(file_path, S3_BUCKET, filename)
        file_url = f"https://d38hsgu31iejbq.cloudfront.net/{filename}"
        return file_url
    except FileNotFoundError:
        return "The file was not found"
    except NoCredentialsError:
        return "Credentials not available"

def process_file_basic(file_path: str):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Get the number of rows
    num_rows = len(df)
    
    # Define a mapping for types
    type_mapping = {
        'object': 'string',
        'datetime': 'date',
        'int': 'number',
        'float': 'number'
    }
    
    # Get the fields (column names) and their types
    fields = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        mapped_type = 'string'  # default type
        for key in type_mapping:
            if key in col_type:
                mapped_type = type_mapping[key]
                break
        fields.append({
            'name': col,
            'type': mapped_type
        })
    
    # Get the file size in bytes
    file_size = os.path.getsize(file_path)
    
    return num_rows, fields, file_size

def get_answer_from_csv(question, csv_file_path):
    file_extension = os.path.splitext(csv_file_path)[1].lower()

    try:
        if file_extension == '.csv':
            df = pd.read_csv(csv_file_path, encoding='utf-8')
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(csv_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save the dataframe to a temporary CSV file for processing
        temp_csv_path = csv_file_path + "_temp.csv"
        df.to_csv(temp_csv_path, index=False)

        agent = create_csv_agent(openai, temp_csv_path, verbose=True, allow_dangerous_code=True)
        result = agent.run(question)
        
        # Assuming the last line after "Final Answer: " contains the answer
        answer = {
            "question": question,
            "answer": result.split("Final Answer: ")[-1]
        }

        os.remove(temp_csv_path)

        return answer
    
    except FileNotFoundError:
        print(f"CSV file not found at path: {csv_file_path}")
        return {"error": "CSV file not found"}


    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "Unexpected error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
