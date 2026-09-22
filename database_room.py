import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Union, Tuple
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


from typing import Annotated
from typing_extensions import TypedDict
import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import logging
import backoff
import uuid
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Store active conversation IDs by session ID
active_conversations = {}



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
            # Get dataframe as results, sql query as query_text
            query_text ='''SELECT `DOSESEQP`, AVG(`AGE`) AS `average_age`
                    FROM `patient_profile`
                    GROUP BY `DOSESEQP` HAVING `DOSESEQP` IS NOT NULL
                    '''
            average_age_data = {
                'DOSESEQP': [160, 10, 80, 40],
                'average_age': [50.0714, 49.2619, 44.2778, 41.2308]}
            result = pd.DataFrame(average_age_data)
            
            #result, query_text = continue_conversation(conversation_id, question)
            
            # If we got an error about conversation not found, start a new one
            if isinstance(result, str) and "conversation has expired" in result:
                logger.info(f"Conversation {conversation_id} expired, starting new one")
                conversation_id='0f123456789'
                query_text ='''SELECT `DOSESEQP`, AVG(`AGE`) AS `average_age`
                    FROM `patient_profile`
                    GROUP BY `DOSESEQP` HAVING `DOSESEQP` IS NOT NULL
                    '''
                average_age_data = {
                    'DOSESEQP': [160, 10, 80, 40],
                    'average_age': [50.0714, 49.2619, 44.2778, 41.2308]}
                result = pd.DataFrame(average_age_data)
                #conversation_id, result, query_text = start_new_conversation(question)
                
                # Update the conversation mapping
                if conversation_id:
                    active_conversations[session_id] = conversation_id
            
            return result, query_text
            
        except Exception as e:
            logger.error(f"Error in existing conversation: {str(e)}")
            return f"Sorry, an error occurred: {str(e)}", None
    else:
        # No existing conversation, start a new one
        conversation_id='0f123456789'
        query_text ='''SELECT `DOSESEQP`, AVG(`AGE`) AS `average_age`
            FROM `patient_profile`
            GROUP BY `DOSESEQP` HAVING `DOSESEQP` IS NOT NULL
            '''
        average_age_data = {
            'DOSESEQP': [160, 10, 80, 40],
            'average_age': [50.0714, 49.2619, 44.2778, 41.2308]}
        result = pd.DataFrame(average_age_data)
        #conversation_id, result, query_text = start_new_conversation(question)
        
        # Store the conversation ID for this session
        if conversation_id:
            active_conversations[session_id] = conversation_id
            
        return result, query_text
    
#============Start here===============
# Please put the API key here

OPEN_AI_KEY = ""

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

#---------- PROMPTS ----------#
AGENT_SYS_PROMP = """
Role: You are biomarker assistant tasked with providing up-to-date information about the clinical trial data.

Objective: Assist data-driven researchers by giving accurate information about the clinical trial data.

Capabilities: You have been given a tool called get_data_from_database. Please retrieve your data only from this tool. Do not construct any query.

Next Steps: When asked to visualize data, provide Python code using Plotly Express to generate charts. 
The plotting figure sizes should be 5 by 3. Select different colours and styles.
Include all necessary imports in your code snippets.
"""



@tool
def get_data_from_database(question: str, session_id: str):
    """Retrieve data from database on the prompt request"""

    query_description = '''This analysis provides the average age of participants,
    categorized by their dose sequence. The results focus on groups
    where the dose sequence is specified, allowing for insights into age distribution across different dosing protocols.
    '''
    response, query_text = genie_query(question, session_id)
    sql_query = f"\n```SQL\n{query_text}\n```\n\n"
    
    output = (query_description + """:\n\n """ +
        response.to_markdown() +
        """ \n\n """ + sql_query)
    return output


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

def setup_agent():
    """Set up and return the LangGraph agent for text-to-SQL queries"""
    graph_builder = StateGraph(GraphState)
    
    tools = [get_data_from_database]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    SYS_MSG = SystemMessage(content=AGENT_SYS_PROMP)
    
    def chatbot(state: GraphState):
        return {"messages": [llm_with_tools.invoke([SYS_MSG] + state["messages"])]}
    
    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node('tools', tool_node)
    graph_builder.add_conditional_edges('chatbot',
                                        tools_condition,
                                        ['tools', END])
    graph_builder.add_edge('tools', 'chatbot')
    graph_builder.set_entry_point('chatbot')
    
   
    
    return graph_builder.compile()

