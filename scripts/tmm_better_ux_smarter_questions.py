# ## Code

# %% [markdown]
# ### Imports

# %%
import boto3
import json
import os
from abc import ABC, abstractmethod
import yaml
from pprint import pprint
from datetime import datetime

# %%
# Constants
YAML_QUESTIONS_PATH = 'Python/data/config_with_desc.yaml'
CONVERSATION_PROMPT_PATH = 'Python/data/claude_triage_prompt.md'
CLAUDE_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
NOVA_MODEL_ID = "us.amazon.nova-premier-v1:0"
NUM_CONVO_LOOPS = 10

# Get questions from YAML
with open(YAML_QUESTIONS_PATH, 'r') as f:
    QUESTIONS_YAML = f.read()

# Get convo prompt from .md
with open(CONVERSATION_PROMPT_PATH, 'r',encoding='utf-8') as f:
    CONVERSATION_PROMPT_TEMPLATE = f.read()

# prompt template adjustments
CONVERSATION_PROMPT = CONVERSATION_PROMPT_TEMPLATE \
    .replace('{QUESTIONS_YAML}', QUESTIONS_YAML) \
    .replace('{NUM_CONVO_LOOPS}', str(NUM_CONVO_LOOPS))

# %%
# Clients
bedrock = boto3.client('bedrock-runtime')
bedrock_agent = boto3.client('bedrock-agent-runtime')

# %% [markdown]
# ### Model Classes

# %%
# Abstract class for model input objects
class ModelCall(ABC):

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

# %%
# Class for Claude models
class ClaudeModelCall(ModelCall):

    def get_model_input(self, system_prompt: str, user_message: str, **kwargs) -> dict:

        # process optional kwargs
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

        # Add provided kwargs
        if (temp != -1):
            model_input_body['temperature'] = temp

        return model_input_body

    def get_model_output(self, model_input: dict) -> str:
        
        # Make the LLM call
        response = bedrock.invoke_model(
            body = json.dumps(model_input), 
            contentType = 'application/json', 
            modelId = CLAUDE_MODEL_ID, 
        )
        model_response_text = json.loads(response["body"].read())['content'][0]['text']

        return model_response_text

# %%
# Class for Claude models
class NovaModelCall(ModelCall):

    def get_model_input(self, system_prompt: str, user_message: str, **kwargs) -> dict:

        # TODO: process optional kwargs
        temp = kwargs.get('temperature', -1)

        user_message_block = [
            {
                "role": "user", 
                'content': [
                    {
                        "text": user_message
                    }
                ]
            }
        ]

        # Format input
        model_input_body = {
            "system": [
                {
                    "text": system_prompt
                }
            ], 
            "messages": user_message_block
        }

        # Add provided kwargs
        if (temp != -1):
            if ('inferenceConfig' not in model_input_body):
                model_input_body['inferenceConfig'] = {}
            model_input_body['inferenceConfig']['temperature'] = temp

        return model_input_body

    def get_model_output(self, model_input: dict) -> str:

        # Make the LLM call
        response = bedrock.invoke_model(
            body = json.dumps(model_input), 
            contentType = 'application/json', 
            modelId = NOVA_MODEL_ID, 
        )
        model_response_text = json.loads(response["body"].read())['output']['message']['content'][0]['text']

        return model_response_text

# %% [markdown]
# ### Session Management

# %%
def get_chat_history(session_id: str) -> str:

    # List invocations in session
    list_invc_resp = bedrock_agent.list_invocations(
        maxResults = 5, 
        sessionIdentifier = session_id
    )
    invc_id_list = [invc_obj['invocationId'] for invc_obj in list_invc_resp['invocationSummaries']]

    # For each invocation, extract text
    answered_questions = {}
    chat_hist_list = []
    invc_obj = None
    for invc_id in reversed(invc_id_list):
        for invc_step in bedrock_agent.list_invocation_steps(invocationIdentifier = invc_id, sessionIdentifier = session_id)['invocationStepSummaries']:
            
            invc_step_id = invc_step['invocationStepId']
            invc_obj = bedrock_agent.get_invocation_step(invocationIdentifier = invc_id, invocationStepId = invc_step_id, sessionIdentifier = session_id)['invocationStep']
            
            # Parse each invocation step
            # Add chat history to list, answered questions to dict
            invc_obj = json.loads(invc_obj['payload']['contentBlocks'][0]['text'])
            chat_hist_list.append(invc_obj['conversation'])
    
    # Get answers dict from the latest invc obj
    if (invc_obj):
        answered_questions = invc_obj['answered_questions']

    # Pass all this to history assembler
    # assembled_hist = '\n\n'.join(invc_text_list)

    # Return assembled history
    return {
        'answered_questions': answered_questions, 
        'chat_history': '\n\n'.join(chat_hist_list)
    }

# %%
def record_invocation(session_id: str, invocation_text: str,  timestamp: datetime):

    # invocation_step must be a json object with 'conversation' and 'answered_questions'

    # Create invocation
    invc_id = bedrock_agent.create_invocation(sessionIdentifier = session_id)['invocationId']

    # Create and record invocation steps
    invc_step_resp = bedrock_agent.put_invocation_step(
        invocationIdentifier = invc_id, 
        invocationStepTime = timestamp, 
        payload = {
            'contentBlocks': [
                {
                    'text': invocation_text
                }
            ]
        }, 
        sessionIdentifier = session_id
    )

    # return invocation and step ID
    return invc_id, invc_step_resp['invocationStepId']

# %% [markdown]
# ### LLM functions

# %%
def evaluate_and_follow_up(
        question: str, 
        answer: str, 
        iter_number: int, 
        model_call_obj: ModelCall, 
        chat_history: str, 
        answers: str):
    
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

    raw_model_response_text = model_call_obj.ask_model(CONVERSATION_PROMPT, user_prompt, temperature = 1)

    # Clean up model response string
    cleaned = raw_model_response_text.replace('```json\n', '').replace('```', '')
    
    return json.loads(cleaned)

# %%
def generate_summary(answers: str, model_call_obj: ModelCall) -> str:

    # Create string with questions and answers

    # Create prompt
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

# %% [markdown]
# ### Main Function

# %%
# Master function
def main(model_name: str = 'claude') -> str:

    # Declare important variables:
    # answers - dict of {'question': 'answer'}
    answers = {}
    # question - string of question(s) to ask - initialize with default first question
    question = 'Please describe your problem'
    are_all_details_captured = False
    chat_history, ans_text = '', ''
    n_loops = 0
    sess_obj = None

    # Resolve model call object
    model_call_dict = {
        'nova': NovaModelCall(), 
        'claude': ClaudeModelCall()
    }
    if (model_name not in model_call_dict): model_name = 'claude'

    # Create a bedrock session for conversation history
    sess_id = bedrock_agent.create_session()['sessionId']

    # start the LLM judging loop - run for a max of 3 times
    while (n_loops < NUM_CONVO_LOOPS and not are_all_details_captured):

        # Ask "question"
        curr_ans = input(question)

        # send the answers, along with the question to the LLM
        # function should return a dict: 
            # "answered_questions" - dict of questions and answers
            # "followup" = string of no more than 5 follow-up questions
        llm_response = evaluate_and_follow_up(question, curr_ans, n_loops, model_call_dict[model_name], chat_history, ans_text)
        
        # Print for debugging
        print (f"===== \nCandidates: {llm_response['candidate_questions']}")
        print (f"Reason: {llm_response['follow_up_reasoning']}")

        # update answers dict with "answered_questions"
        answers.update(llm_response['answered_questions'])

        # Add this entry to bedrock session
        payload = {
            'conversation': f'Assistant - {question}\nUser - {curr_ans}', 
            'answered_questions': answers
        }
        invc_id, step_id = record_invocation(sess_id, json.dumps(payload), datetime.now())

        # Set question to followup generated by LLM
        question = llm_response['followup']

        # See if all questions are answered
        are_all_details_captured = llm_response['are_all_details_captured']
        
        # Get chat history and answers from session management
        sess_obj = get_chat_history(sess_id)
        chat_history = sess_obj['chat_history']
        ans_text = '\n\n'.join([f'{q}\n{a}' for q, a in answers.items()])

        print (chat_history)

        # increment n_loops by 1
        n_loops += 1

    # For last question, always ask for additional details
    question = 'Thanks for all the details. Is there anything else you\'d like our technicians to know?'
    curr_ans = input(question)
    answers[question] = curr_ans
    ans_text = f'{ans_text}\n\n{question}\n{curr_ans}'

    # Close bedrock session
    end_sess_resp = bedrock_agent.end_session(sessionIdentifier = sess_id)

    # Generate one line summary
    summary = generate_summary(ans_text, model_call_dict[model_name])

    # Final message to user
    print (f'Thank you for your responses. A repair order has been submitted and a technician will look into it at the earliest. Here\'s the summary of your issues:\n{summary}')
    
    return answers, summary

# %%
answers, summary = main(model_name = 'claude')
