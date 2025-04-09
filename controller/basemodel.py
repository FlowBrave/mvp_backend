from pydantic import BaseModel

    
class CreateAgentRequest(BaseModel):
    agent_name: str
    prompt: str
    user_email: str
    search_type: str

class ConversationRequest(BaseModel):
    agent_id: int
    query: str