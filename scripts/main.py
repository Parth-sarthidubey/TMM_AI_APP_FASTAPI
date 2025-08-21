import boto3
import json
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import uvicorn
import io

# --- Constants and Boto3 Clients ---
# Ensure these paths are correct relative to where you run the server
try:
    with open('../data/config_with_desc.yaml', 'r') as f:
        QUESTIONS_YAML = f.read()
    with open('../data/claude_triage_prompt.md', 'r', encoding='UTF-8') as f:
        CONVERSATION_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    print("Warning: Could not find YAML or MD files. Using placeholder content.")
    QUESTIONS_YAML = "{}"
    CONVERSATION_PROMPT_TEMPLATE = "Prompt template"


CLAUDE_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
NUM_CONVO_LOOPS = 10

CONVERSATION_PROMPT = CONVERSATION_PROMPT_TEMPLATE \
    .replace('{QUESTIONS_YAML}', QUESTIONS_YAML) \
    .replace('{NUM_CONVO_LOOPS}', str(NUM_CONVO_LOOPS))

# AWS Clients
bedrock = boto3.client('bedrock-runtime')

try:
    polly_client = boto3.client('polly')
except Exception as e:
    print("Error initializing Boto3 Polly client:", e)
    polly_client = None

# Pydantic model for validating the incoming request data
class SynthesizeRequest(BaseModel):
    text: str

# --- Model Classes (Directly from your script) ---
class ModelCall:
    @abstractmethod
    def get_model_input(self, system_prompt: str, user_message: str, **kwargs) -> dict:
        pass

    @abstractmethod
    def get_model_output(self, model_input: dict) -> str:
        pass

    def ask_model(self, system_prompt: str, user_message: str, **kwargs) -> str:
        return self.get_model_output(
            self.get_model_input(
                system_prompt,
                user_message,
                **kwargs
            )
        )

class ClaudeModelCall(ModelCall):
    def get_model_input(self, system_prompt: str, user_message: str, **kwargs) -> dict:
        max_tokens = kwargs.get('max_tokens', 1024)
        temp = kwargs.get('temperature', -1)
        model_input_body = {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': f'{user_message}'
                        }
                    ]
                }
            ]
        }
        if temp != -1:
            model_input_body['temperature'] = temp
        return model_input_body

    def get_model_output(self, model_input: dict) -> str:
        response = bedrock.invoke_model(
            body=json.dumps(model_input),
            contentType='application/json',
            modelId=CLAUDE_MODEL_ID,
        )
        return json.loads(response["body"].read())['content'][0]['text']

claude_model = ClaudeModelCall()

# --- LLM Functions (Directly from your script) ---
def evaluate_and_follow_up(question: str, answer: str, iter_number: int, model_call_obj: ModelCall, chat_history: str, answers: str):
    user_prompt = f"""
    ## Full conversation
    {chat_history}

    {question}
    {answer}

    ## Extracted details so far
    {answers}

    ## Current number of responses
    {iter_number}
    """
    raw_model_response_text = model_call_obj.ask_model(CONVERSATION_PROMPT, user_prompt, temperature=1)
    cleaned = raw_model_response_text.replace('```json\n', '').replace('```', '')
    return json.loads(cleaned)

def generate_summary(answers: str, model_call_obj: ModelCall) -> str:
    prompt = f'''
        # Expert Car Dealership Q&A Summarizer

        ## Task
        You are a professional summarizer specializing in customer interactions at car dealerships. Your task is to create a concise summary that captures all critical information from the following customer Q&A session.

        ## Summary Requirements
        - Create an extremely brief summary (maximum 150 words)
        - Use clear, professional language
        - Report only customer observations **without** technical diagnosis
        - Pay special attention to categorized observations (marked by "Which category best describes your observation")
        - Include only information explicitly stated in the conversation
        - Provide only the final summary with no additional text
        - You may extend the summary to 2 or 3 sentences if required.
        - The summary should be in lucid, natural language. **Do not** repeat information in your summary
        - Make sure the summary captures the answers for the questions below. You may omit questions from your summary that don't have an answer from the user
        - Some questions have subcategories, these should also be mentioned in the summary.
       
        ## Questions list
        {QUESTIONS_YAML}

        ## Input format
        - The input is a dictionary with the entire chat session with the user of the format {{"question": "answer"}}

        ## Output Format
        Provide only your summary without any preamble, explanations, or additional commentary. Focus on being concise while capturing the key customer observations, especially the categorized ones.
       '''
    model_response_text = model_call_obj.ask_model(prompt, answers)

    return model_response_text
   

   

# --- Pydantic Models for API Data Validation ---
class ConversationState(BaseModel):
    answers: Dict[str, str] = {}
    chat_history: List[Dict[str, Any]] = []
    loop_count: int = 0
    are_all_details_captured: bool = False
    is_pending_closure: bool = False

class ConversationTurnRequest(BaseModel):
    current_answer: str
    last_question: str
    state: ConversationState

class ConversationTurnResponse(BaseModel):
    next_question: str
    state: ConversationState
    summary: Union[Dict[str, Any], None] = None


# --- FastAPI App Initialization ---
app = FastAPI(title="Stateful Car Repair Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your frontend (e.g., http://localhost:3000) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints for Conversational Loop ---
@app.get("/api/conversation/start")
async def start_conversation():
    """Initializes a new conversation."""
    initial_state = ConversationState()
    first_question = "Hello! I'm the Nissan AI Repair Assistant. How can I help you with your vehicle today?"
    return {"next_question": first_question, "state": initial_state.dict()}

@app.post("/api/conversation/next", response_model=ConversationTurnResponse, response_model_exclude_none=True)
async def handle_conversation_turn(payload: ConversationTurnRequest):
    """Processes one turn of the conversation."""
    state = payload.state

    # 1. Update state
    state.answers[payload.last_question] = payload.current_answer
    state.chat_history.append({'sender': 'bot', 'text': payload.last_question})
    state.chat_history.append({'sender': 'user', 'text': payload.current_answer})
    state.loop_count += 1
   
    chat_history_str = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in state.chat_history])
    answers_str = json.dumps(state.answers, indent=2)

    try:
        llm_response = evaluate_and_follow_up(
            question=payload.last_question,
            answer=payload.current_answer,
            iter_number=state.loop_count,
            model_call_obj=claude_model,
            chat_history=chat_history_str,
            answers=answers_str
        )
        print("LLM response: ",llm_response)
    
        answered_questions=llm_response.get('answered_questions', {})
        for question, answer in answered_questions.items():
        # If the answer is a list with one item, convert it to a string
            if isinstance(answer, list) and len(answer) > 0:
                state.answers[question] = str(answer[0])
            else:
                state.answers[question] = str(answer) # Ensure it's a string

        state.are_all_details_captured = llm_response.get('are_all_details_captured', False)
        next_question = llm_response.get('followup', "Is there anything else you'd like to add?")

    except json.JSONDecodeError:
        next_question = "I had a little trouble processing that. Could you please rephrase?"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {e}")

    # 2. Check for end conditions
    gen_summary=False
    if state.is_pending_closure:
        if 'no' in payload.current_answer.lower():
            gen_summary = True
        else:
            state.answers["Final User Addition"] = payload.current_answer
            gen_summary = True
    
    elif (state.are_all_details_captured or state.loop_count >= NUM_CONVO_LOOPS) and not state.is_pending_closure:
        print("AI believes conversation is over. Asking for final confirmation.")
        state.is_pending_closure = True # Set the flag that we are now pending closure
        return ConversationTurnResponse(
            next_question="I believe I have all the details I need. Is there anything else you would like to add before I create the summary?",
            state=state
        )
    
    if gen_summary:
        final_answers_text = '\n\n'.join([f'Q: {q}\nA: {a}' for q, a in state.answers.items()])
        summary_text = generate_summary(final_answers_text, claude_model)
        
        summary_obj = {
            "issue": summary_text,
            "appointment": "An appointment needs to be scheduled.",
        }
        
        return ConversationTurnResponse(
            next_question="Thank you for confirming. A repair order will be submitted.",
            state=state,
            summary=summary_obj
        )
    
   
   
    # 3. Continue conversation
    return ConversationTurnResponse(next_question=next_question, state=state)

try:
    polly_client = boto3.client('polly')
except Exception as e:
    print("Error initializing Boto3 client. Make sure your AWS credentials are set.")
    print(e)
    polly_client = None

@app.post("/synthesize-speech")
async def synthesize_speech(request: SynthesizeRequest):
    """
    Endpoint to convert text to speech using AWS Polly.
    Accepts a JSON body with a "text" field.
    Returns an MP3 audio stream.
    """
    if not polly_client:
        raise HTTPException(
            status_code=500,
            detail="AWS Polly client not configured on server"
        )

    try:
        # Synthesize speech using a natural-sounding neural voice
        response = polly_client.synthesize_speech(
            Text=request.text,
            OutputFormat='mp3',
            VoiceId='Ruth',      # A popular, natural-sounding voice
            Engine='neural'      # Use the higher quality neural engine
        )
    except Exception as e:
        print(f"Error calling AWS Polly: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to synthesize speech"
        )

    # Stream the audio back to the client
    if 'AudioStream' in response:
        # FastAPI's StreamingResponse is the equivalent of Flask's Response
        audio_stream = response['AudioStream']
        return StreamingResponse(io.BytesIO(audio_stream.read()), media_type='audio/mpeg')

    raise HTTPException(
        status_code=500,
        detail="AudioStream not found in Polly response"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
