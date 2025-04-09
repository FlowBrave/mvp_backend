import pickle
from typing import List,Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Depends,Form
from sqlalchemy.orm import Session
from agents import Agent as AgentClass, WebSearchTool, Runner,function_tool
from openai import OpenAI
import os
from .basemodel import ConversationRequest
from dotenv import load_dotenv
import tempfile
from database import get_db
from model import Agent,Userfile,GeneratedFile
from fpdf import FPDF
import uuid
import base64
from mistralai import Mistral
from pathlib import Path
from docx import Document
import pypandoc
import shutil
import boto3
from pinecone import Pinecone,ServerlessSpec
from botocore.exceptions import NoCredentialsError
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # ✅ Make sure this stays a string

vector_store_id = os.getenv("PINECONE_INDEX_NAME")  # Hardcoded to your index


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "API Key") 
BASE_URL = os.getenv("BASE_URL") 
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")



index = pc.Index(PINECONE_INDEX_NAME)  # ✅ This is the actual index object
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' created.")
else:
    print(f"🔄 Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


app = FastAPI()
router = APIRouter()


UPLOAD_DIR = tempfile.gettempdir()

PDF_STORAGE_PATH = tempfile.mkdtemp()
    
    
    
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)   
    
@function_tool  
def generate_pdf(context: str) -> dict:
    """Generates a PDF document from given text content."""
    pdf_filename = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Generated Document", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, context)
    pdf.output(pdf_path)

    pdf_download_link = f"http://{BASE_URL}/generated_pdf/{pdf_filename}"

    return {
        "message": "PDF generated successfully!", 
        "pdf_download_link": pdf_download_link
    }

@function_tool
def pinecone_search(query: str, agent_id: str) -> str:
    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"agent_id": agent_id}  # 🔒 filter to this agent only
        )

        matches = [m['metadata']['text'] for m in results.matches if 'text' in m.metadata]
        return "\n\n".join(matches) if matches else "No relevant matches found."

    except Exception as e:
        return f"Search error: {str(e)}"

    



  
@router.post("/create-agent")
async def create_agent_with_files(
    agent_name: str = Form(...),
    prompt: str = Form(...),
    user_email: str = Form(...),
    search_type: str = Form(...),
    existing_file_ids: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    files: Optional[List[UploadFile]] = File(None), 
):
    try:
        tools = []
        if search_type == "web_search":
            tools.append(WebSearchTool())
        elif search_type == "Reasoning":
            pass
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")

        agent_instance = AgentClass(
            name=agent_name,
            instructions=prompt,
            model='gpt-4o',
            tools=tools
        )

        if search_type == "Reasoning":
            agent_instance.instructions += (
                "\n\nYou have access to a tool called `pinecone_search` which lets you "
                "search the user's uploaded documents. Always use this tool when the user "
                "asks about uploaded files, document names, or anything that may be in those documents."
            )

        new_agent = Agent(
            agent_name=agent_name,
            instructions=prompt,
            vector_store_id=vector_store_id,
            agent_object=pickle.dumps([agent_instance]),
            user_email=user_email,
            search_type=search_type,
        )
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)

        if files:
            for file in files:
                ext = os.path.splitext(file.filename)[1].lower()
                original_filename = Path(file.filename).stem

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    file_content = await file.read()
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name

                if ext == ".docx":
                    pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                    temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                    os.rename(tmp_file_path, temp_docx_path)
                    pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)
                    tmp_file_path = pdf_path

                extracted_text = process_file(tmp_file_path, MISTRAL_API_KEY)

                embedding_response = client.embeddings.create(
                    input=extracted_text,
                    model="text-embedding-ada-002"
                )
                embedding = embedding_response.data[0].embedding

                index.upsert([(
                    f"{new_agent.id}-{original_filename}-{str(uuid.uuid4())}",
                    embedding,
                    {
                        "text": extracted_text,
                        "agent_id": str(new_agent.id),
                        "user_email": user_email,
                        "filename": file.filename
                    }
                )])

                s3_key = f"user_files/{user_email}/uploaded_files/{file.filename}"
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=file_content)
                s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                db.add(Userfile(user_email=user_email, file_path=s3_file_path))
                db.commit()

                if os.path.exists(tmp_file_path): os.remove(tmp_file_path)
                if ext == ".docx":
                    if os.path.exists(temp_docx_path): os.remove(temp_docx_path)
                    if os.path.exists(pdf_path): os.remove(pdf_path)

        # ✅ 2. Handle selected files from document repository
        if existing_file_ids:
            file_ids = [int(fid.strip()) for fid in existing_file_ids.split(",") if fid.strip().isdigit()]
            for file_id in file_ids:
                file_record = db.query(Userfile).filter(Userfile.id == file_id, Userfile.user_email == user_email).first()
                if not file_record:
                    print(f"⚠️ File ID {file_id} not found or doesn't belong to user.")
                    continue

                s3_key = file_record.file_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
                file_extension = os.path.splitext(s3_key)[1].lower()
                filename = os.path.basename(s3_key)
                original_filename = Path(filename).stem

                temp_file_path = os.path.join(tempfile.gettempdir(), filename)
                s3_client.download_file(S3_BUCKET_NAME, s3_key, temp_file_path)

                if file_extension == ".docx":
                    pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                    temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                    shutil.copyfile(temp_file_path, temp_docx_path)
                    pypandoc.convert_file(temp_docx_path, to='pdf', outputfile=pdf_path)
                    temp_file_path = pdf_path

                extracted_text = process_file(temp_file_path, MISTRAL_API_KEY)

                embedding_response = client.embeddings.create(
                    input=extracted_text,
                    model="text-embedding-ada-002"
                )
                embedding = embedding_response.data[0].embedding

                index.upsert([(
                    f"{new_agent.id}-{original_filename}-{str(uuid.uuid4())}",
                    embedding,
                    {
                        "text": extracted_text,
                        "agent_id": str(new_agent.id),
                        "user_email": user_email,
                        "filename": filename
                    }
                )])

                for path in [temp_file_path, temp_docx_path if file_extension == ".docx" else None]:
                    if path and os.path.exists(path): os.remove(path)

        return {
            "message": "Agent created and files (if any) embedded successfully",
            "agent_id": new_agent.id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")
    
    
@router.get("/get-agents")
def get_agents(user_email: str, db: Session = Depends(get_db)):
    try:
        agents = (
            db.query(Agent)
            .filter(Agent.user_email == user_email)
            .order_by(Agent.id.desc())  # sort: newest first
            .all()
        )

        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.agent_name,
                    "description": agent.instructions,
                    "search_type": agent.search_type
                }
                for agent in agents
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching agents: {str(e)}")
    
  
@router.delete("/delete-agent/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Looking for agent with ID: {agent_id}")
        agent = db.query(Agent).filter(Agent.id == agent_id).first()

        if not agent:
            raise HTTPException(status_code=404, detail="❌ Agent not found")

        print(f"🧠 Deleting agent: {agent.agent_name} (search_type: {agent.search_type})")

        # Delete associated embeddings from Pinecone
        print(f"🧹 Deleting vectors from Pinecone for agent_id={agent_id}")
        try:
            index.delete(filter={"agent_id": str(agent_id)})
            print("✅ Pinecone embeddings deleted successfully.")
        except Exception as ve:
            print(f"⚠️ Warning: Failed to delete Pinecone vectors - {ve}")

        # Delete agent from DB
        db.delete(agent)
        db.commit()
        print("🗃️ Agent deleted from database.")

        return {"message": "✅ Agent and associated vector embeddings deleted successfully"}

    except Exception as e:
        print(f"❌ Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
    

@router.post("/conversation")
async def conversation(request: ConversationRequest, db: Session = Depends(get_db)):
    try:
        agent_data = db.query(Agent).filter(Agent.id == request.agent_id).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        user_email = agent_data.user_email 

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]
        
        agent_instance.tools.append(generate_pdf)
        print(agent_instance)
        
        if agent_data.search_type == "Reasoning":
            agent_instance.tools.append(pinecone_search)
            agent_instance.instructions += (
                f"\n\nYou have access to a tool called `pinecone_search` which lets you "
                f"search the user's uploaded documents. Always use this tool when the user "
                f"asks about uploaded files, document names, or anything that may be in those documents. "
                f"The current agent ID is {request.agent_id}. Use it in the search."
            )
            print(agent_instance)

        result = await Runner.run(agent_instance, request.query)
        response_text = result.final_output
        import re
        match = re.search(r"http://localhost:8000/generated_pdf/([a-zA-Z0-9\-_]+\.pdf)", response_text)
        if match:
            filename = match.group(1)
            local_pdf_path = os.path.join(PDF_STORAGE_PATH, filename)

            if os.path.exists(local_pdf_path):
                s3_key = f"user_files/{user_email}/generated_files/{filename}"
                with open(local_pdf_path, "rb") as f:
                    s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)

                s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"

                new_file = GeneratedFile(user_email=user_email, file_path=s3_file_path)
                db.add(new_file)
                db.commit()
                db.refresh(new_file)

                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
                    ExpiresIn=3600
                )

                os.remove(local_pdf_path)

                return {
                    "message": "PDF generated and uploaded to S3",
                    "file_id": new_file.id,
                    "download_url": presigned_url,
                    "response": response_text
                }

        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")
          


def process_file(file_path, api_key):
    client = Mistral(api_key=api_key)

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    with open(file_path, 'rb') as file:
        file_bytes = file.read()

    if file_extension in ['.pdf']:
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_file}"}
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }.get(file_extension)
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")
        document = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_file}"}
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document)
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else (ocr_response if isinstance(ocr_response, list) else [])
        result_text = "\n\n".join(page.markdown for page in pages) or "No result found."
        print(result_text)

        return result_text
    except Exception as e:
        return f"Error extracting text: {e}"
    

    
@router.post("/conversation-file1")
   
async def conversation_file(
    agent_id: int = Form(...),
    query: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    db: Session = Depends(get_db)
):
    try:
        agent_data = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]

        agent_instance.tools.append(generate_pdf)

        if agent_data.search_type == "Reasoning" or files:
            agent_instance.tools.append(pinecone_search)
            agent_instance.instructions += (
                "\n\nYou have access to a tool called `pinecone_search` which lets you "
                "search the user's uploaded documents. Always use this tool when the user "
                "asks about uploaded files, document names, or anything that may be in those documents. "
                f"The current agent ID is {agent_id}. Use it in the search."
            )

        uploaded_files_info = []

        if files:
            for file in files:
                file_extension = os.path.splitext(file.filename)[1].lower()
                original_filename = Path(file.filename).stem

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file_path = tmp_file.name
                    file_content = await file.read()
                    tmp_file.write(file_content)

                if file_extension == ".docx":
                    pdf_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.pdf")
                    temp_docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}_copy.docx")
                    os.rename(tmp_file_path, temp_docx_path)
                    pypandoc.convert_file(temp_docx_path, to="pdf", outputfile=pdf_path)
                    tmp_file_path = pdf_path
                    file_extension = ".pdf"

                extracted_text = process_file(tmp_file_path, MISTRAL_API_KEY)

                docx_path = os.path.join(tempfile.gettempdir(), f"{original_filename}.docx")
                doc = Document()
                doc.add_paragraph(extracted_text)
                doc.save(docx_path)

                embedding_response = client.embeddings.create(
                    input=extracted_text,
                    model="text-embedding-ada-002"
                )
                embedding = embedding_response.data[0].embedding

                index.upsert([(
                    f"{agent_id}-{original_filename}-{str(uuid.uuid4())}",
                    embedding,
                    {
                        "text": extracted_text,
                        "agent_id": str(agent_id),
                        "user_email": agent_data.user_email,
                        "filename": file.filename
                    }
                )])

                # Upload to S3
                s3_key = f"user_files/{agent_data.user_email}/uploaded/{file.filename}"
                with open(tmp_file_path, "rb") as f:
                    s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)

                s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
                new_file = Userfile(user_email=agent_data.user_email, file_path=s3_file_path)
                db.add(new_file)
                db.commit()
                db.refresh(new_file)

                uploaded_files_info.append({
                    "file_name": file.filename,
                    "file_s3_path": s3_file_path
                })

                for path in [tmp_file_path, docx_path, temp_docx_path if file_extension == ".docx" else None]:
                    if path and os.path.exists(path):
                        os.remove(path)

        # Run agent on the query after all files processed
        result = await Runner.run(agent_instance, query)

        return {
            "response": result.final_output,
            "files_uploaded": uploaded_files_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")
    
    
@router.post("/upload-file/")
async def upload_user_files(
    user_email: str = Form(...),
    folder_path: str = Form(None),
    files: List[UploadFile] = File(...),  # ✅ multiple files
    db: Session = Depends(get_db)
):
    try:
        folder_path = folder_path.strip("/") if folder_path else "uploaded_files"
        uploaded_files_info = []

        for file in files:
            file_content = await file.read()
            s3_key = f"user_files/{user_email}/{folder_path}/{file.filename}"
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=file_content)
            s3_file_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"

            file_record = Userfile(user_email=user_email, file_path=s3_file_path)
            db.add(file_record)
            db.commit()
            db.refresh(file_record)

            uploaded_files_info.append({
                "file_id": file_record.id,
                "file_path": s3_file_path,
                "file_name": file.filename
            })

        return {
            "message": "✅ Files uploaded successfully!",
            "files": uploaded_files_info
        }

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
@router.get("/get-user-files/")
def get_user_files(user_email: str, db: Session = Depends(get_db)):
    try:
        files = db.query(Userfile).filter(Userfile.user_email == user_email).all()
        presigned_files = []

        for file in files:
            file_key = file.file_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET_NAME, "Key": file_key},
                ExpiresIn=3600  # 1 hour
            )
            presigned_files.append({
                "file_id": file.id,
                "file_path": file.file_path,
                "uploaded_at": file.uploaded_at,
                "presigned_url": presigned_url
            })

        return {
            "user_email": user_email,
            "total_files": len(presigned_files),
            "files": presigned_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")

    
@router.delete("/delete-file/{file_id}")
def delete_user_file(file_id: int, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Looking for file with ID: {file_id}")
        file_record = db.query(Userfile).filter(Userfile.id == file_id).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="❌ File not found")

        full_s3_path = file_record.file_path
        print(f"📦 Full S3 path from DB: {full_s3_path}")

        if not full_s3_path.startswith(f"s3://{S3_BUCKET_NAME}/"):
            raise HTTPException(status_code=400, detail="⚠️ Invalid S3 file path format")

        s3_key = full_s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
        print(f"🗝️ Extracted S3 key: {s3_key}")

        # Delete from S3
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        print(f"✅ File deleted from S3 bucket: {S3_BUCKET_NAME}")

        # Delete from DB
        db.delete(file_record)
        db.commit()
        print(f"🗃️ File record deleted from database")

        return {"message": "✅ File deleted successfully from S3 and database"}

    except Exception as e:
        print(f"❌ Deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

@router.get("/get-genrated-files/")
def get_user_files(user_email: str, db: Session = Depends(get_db)):
    try:


        files = db.query(GeneratedFile).filter(GeneratedFile.user_email == user_email).all()

        return {
            "user_email": user_email,
            "total_files": len(files),
            "files": [
                {
                    "file_id": file.id,
                    "file_path": file.file_path
                }
                for file in files
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching files: {str(e)}")
    


@router.put("/edit-agent/{agent_id}")
async def edit_agent(
    agent_id: int,
    user_email: str = Form(...),
    new_name: str = Form(...),
    new_instructions: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        agent_data = db.query(Agent).filter(Agent.id == agent_id, Agent.user_email == user_email).first()
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found or does not belong to user")

        agent_data.agent_name = new_name
        agent_data.instructions = new_instructions

        agent_list = pickle.loads(agent_data.agent_object)
        agent_instance = agent_list[0]
        agent_instance.name = new_name
        agent_instance.instructions = new_instructions

        agent_data.agent_object = pickle.dumps([agent_instance])
        db.commit()


        return {"message": "Agent updated successfully with new files and embeddings"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error editing agent: {str(e)}")
    
    
