import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import backoff
import uuid
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load environment variables
SPACE_ID = os.environ.get("SPACE_ID")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

# Store active conversation IDs by session ID
active_conversations = {}

class GenieClient:
    def __init__(self, host: str, token: str, space_id: str):
        self.host = host
        self.token = token
        self.space_id = space_id
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.base_url = f"https://{host}/api/2.0/genie/spaces/{space_id}"
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def start_conversation(self, question: str) -> Dict[str, Any]:
        """Start a new conversation with the given question"""
        url = f"{self.base_url}/start-conversation"
        payload = {"content": question}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send a follow-up message to an existing conversation"""
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": message}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_message(self, conversation_id: str, message_id: str) -> Dict[str, Any]:
        """Get the details of a specific message"""
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def get_query_result(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Get the query result using the attachment_id endpoint"""
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        
        # Extract data_array from the correct nested location
        data_array = []
        if 'statement_response' in result:
            if 'result' in result['statement_response']:
                data_array = result['statement_response']['result'].get('data_array', [])
            
        return {
                    'data_array': data_array,
                    'schema': result.get('statement_response', {}).get('manifest', {}).get('schema', {})
                }

    @backoff.on_exception(
        backoff.expo,
        Exception,  # Retry on any exception
        max_tries=5,
        factor=2,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"API request failed. Retrying in {details['wait']:.2f} seconds (attempt {details['tries']})"
        )
    )
    def execute_query(self, conversation_id: str, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Execute a query using the attachment_id endpoint"""
        url = f"{self.base_url}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/execute-query"
        
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    

    def wait_for_message_completion(self, conversation_id: str, message_id: str, timeout: int = 300, poll_interval: int = 2) -> Dict[str, Any]:
        """
        Wait for a message to reach a terminal state (COMPLETED, ERROR, etc.).
        
        Args:
            conversation_id: The ID of the conversation
            message_id: The ID of the message
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            
        Returns:
            The completed message
        """
        
        start_time = time.time()
        attempt = 1
        
        while time.time() - start_time < timeout:
            
            message = self.get_message(conversation_id, message_id)
            status = message.get("status")
            
            if status in ["COMPLETED", "ERROR", "FAILED"]:
                return message
                
            time.sleep(poll_interval)
            attempt += 1
            
        raise TimeoutError(f"Message processing timed out after {timeout} seconds")

def start_new_conversation(question: str) -> Tuple[str, Union[str, pd.DataFrame], Optional[str]]:
    """
    Start a new conversation with Genie.
    
    Args:
        question: The initial question
        
    Returns:
        Tuple containing:
        - conversation_id: The new conversation ID
        - response: Either text or DataFrame response
        - query_text: SQL query text if applicable, otherwise None
    """
    logger.info(f"Starting new conversation with question: {question[:30]}...")
    
    client = GenieClient(
        host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        space_id=SPACE_ID
    )
    
    try:
        # Start a new conversation
        response = client.start_conversation(question)
        conversation_id = response.get("conversation_id")
        message_id = response.get("message_id")
        
        logger.info(f"Created new conversation {conversation_id}")
        
        # Wait for the message to complete
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        
        # Process the response
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        
        return conversation_id, result, query_text
        
    except Exception as e:
        logger.error(f"Error starting new conversation: {str(e)}")
        return None, f"Sorry, an error occurred: {str(e)}", None

def continue_conversation(conversation_id: str, question: str) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    """
    Send a follow-up message in an existing conversation.
    
    Args:
        conversation_id: The existing conversation ID
        question: The follow-up question
        
    Returns:
        Tuple containing:
        - response: Either text or DataFrame response
        - query_text: SQL query text if applicable, otherwise None
    """
    logger.info(f"Continuing conversation {conversation_id} with question: {question[:30]}...")
    
    client = GenieClient(
        host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        space_id=SPACE_ID
    )
    
    try:
        # Send follow-up message in existing conversation
        response = client.send_message(conversation_id, question)
        message_id = response.get("message_id")
        
        # Wait for the message to complete
        complete_message = client.wait_for_message_completion(conversation_id, message_id)
        
        # Process the response
        result, query_text = process_genie_response(client, conversation_id, message_id, complete_message)
        
        return result, query_text
        
    except Exception as e:
        # Handle specific errors
        if "429" in str(e) or "Too Many Requests" in str(e):
            return "Sorry, the system is currently experiencing high demand. Please try again in a few moments.", None
        elif "Conversation not found" in str(e):
            return "Sorry, the previous conversation has expired. Please try your query again to start a new conversation.", None
        else:
            logger.error(f"Error continuing conversation: {str(e)}")
            return f"Sorry, an error occurred: {str(e)}", None

def process_genie_response(client, conversation_id, message_id, complete_message) -> Tuple[Union[str, pd.DataFrame], Optional[str]]:
    """
    Process the response from Genie
    
    Args:
        client: The GenieClient instance
        conversation_id: The conversation ID
        message_id: The message ID
        complete_message: The completed message response
        
    Returns:
        Tuple containing:
        - result: Either text or DataFrame response
        - query_text: SQL query text if applicable, otherwise None
    """
    # Check attachments first
    attachments = complete_message.get("attachments", [])
    for attachment in attachments:
        attachment_id = attachment.get("attachment_id")
        
        # If there's text content in the attachment, return it
        if "text" in attachment and "content" in attachment["text"]:
            return attachment["text"]["content"], None
        
        # If there's a query, get the result
        elif "query" in attachment:
            query_text = attachment.get("query", {}).get("query", "")
            query_result = client.get_query_result(conversation_id, message_id, attachment_id)
           
            data_array = query_result.get('data_array', [])
            schema = query_result.get('schema', {})
            columns = [col.get('name') for col in schema.get('columns', [])]
            
            # If we have data, return as DataFrame
            if data_array:
                # If no columns from schema, create generic ones
                if not columns and data_array and len(data_array) > 0:
                    columns = [f"column_{i}" for i in range(len(data_array[0]))]
                
                df = pd.DataFrame(data_array, columns=columns)
                return df, query_text
    
    # If no attachments or no data in attachments, return text content
    if 'content' in complete_message:
        return complete_message.get('content', ''), None
    
    return "No response available", None

def genie_query(question: str, session_id: str = None) -> Union[Tuple[str, Optional[str]], Tuple[pd.DataFrame, str]]:
    """
    Main entry point for querying Genie.
    
    Args:
        question: The question to ask
        session_id: The Dash session ID to associate with a conversation
        
    Returns:
        Tuple containing either:
        - (text_response, None) for text responses
        - (dataframe, sql_query) for data responses
    """
    global active_conversations
    if not session_id:
        return "No session ID provided", None
    
    logger.info(f"Processing query for session {session_id}: {question[:30]}...")
    
    # Check if we have an existing conversation for this session
    if session_id in active_conversations:
        conversation_id = active_conversations[session_id]
        logger.info(f"Found existing conversation {conversation_id} for session {session_id}")
        
        try:
            # Continue existing conversation
            result, query_text = continue_conversation(conversation_id, question)
            
            # If we got an error about conversation not found, start a new one
            if isinstance(result, str) and "conversation has expired" in result:
                logger.info(f"Conversation {conversation_id} expired, starting new one")
                conversation_id, result, query_text = start_new_conversation(question)
                
                # Update the conversation mapping
                if conversation_id:
                    active_conversations[session_id] = conversation_id
            
            return result, query_text
            
        except Exception as e:
            logger.error(f"Error in existing conversation: {str(e)}")
            return f"Sorry, an error occurred: {str(e)}", None
    else:
        # No existing conversation, start a new one
        conversation_id, result, query_text = start_new_conversation(question)
        
        # Store the conversation ID for this session
        if conversation_id:
            active_conversations[session_id] = conversation_id
            
        return result, query_text

def clear_conversation(session_id: str) -> None:
    """
    Clear the conversation mapping for a session when "New Chat" is clicked.
    
    Args:
        session_id: The session ID to clear
    """
    global active_conversations
    
    if session_id in active_conversations:
        logger.info(f"Clearing conversation for session {session_id}")
        del active_conversations[session_id]

