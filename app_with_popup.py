import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context, no_update, clientside_callback, dash_table
import dash_bootstrap_components as dbc
import json
#from genie_room import genie_query, clear_conversation
from database_room import genie_query, clear_conversation, setup_agent
from langchain_core.messages import SystemMessage, HumanMessage
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
import pandas as pd
import sqlparse
import re, os
import uuid


def get_fig_from_code(code):
    local_variables = {}
    exec(code, {}, local_variables)
    return local_variables['fig']


text2sql_agent = setup_agent()



app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Define the layout
app.layout = html.Div([
    popup_modal,
    # Top navigation bar
    html.Div([
        # Left component containing both nav-left and sidebar
        html.Div([
            # Nav left
            html.Div([
                html.Button([
                    html.Img(src="assets/menu_icon.svg", className="menu-icon")
                ], id="sidebar-toggle", className="nav-button"),
                # Speech icon in top nav (visible when sidebar is closed)
                html.Button([
                    html.Img(src="assets/plus_icon.svg", className="new-chat-icon")
                ], id="new-chat-button", className="nav-button",disabled=False),
                html.Button([
                    html.Img(src="assets/plus_icon.svg", className="new-chat-icon"),
                    html.Div("New chat", className="new-chat-text")
                ], id="sidebar-new-chat-button", className="new-chat-button",disabled=False)
            ], id="nav-left", className="nav-left"),
            
            # Sidebar (now inside the left component)
            html.Div([
                html.Div([
                    html.Div("Your conversations with Genie", className="sidebar-header-text"),
                ], className="sidebar-header"),
                html.Div([], className="chat-list", id="chat-list")
            ], id="sidebar", className="sidebar")
        ], id="left-component", className="left-component"),
        
        html.Div([
            html.Div("Genie Space", id="logo-container", className="logo-container")
        ], className="nav-center"),
        html.Div([
            html.Div("S", className="user-avatar")
        ], className="nav-right")
    ], className="top-nav"),
    
    # Main content area with chat
    html.Div([
        # Main chat area
        html.Div([
            # Chat content
            html.Div([
                # Initial welcome message
                html.Div([
                    html.Div([html.Div([
                    html.Div(className="genie-logo")
                ], className="genie-logo-container")],
                className="genie-logo-container-header"),
               
                    html.Div("Biomedical Clinical Trial", className="welcome-message"),
                
                    html.Div("Analyze Biomedical clinical trials leveraging AI/BI Dashboard. Deep dive into your data and metrics.", 
                             className="welcome-message-description"),
                    
                    # Suggestion buttons with IDs
                    html.Div([
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("What tables are there and how are they connected?")
                        ], id="suggestion-1", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Get average age by dose?")
                        ], id="suggestion-2", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Explain the dataset")
                        ], id="suggestion-3", className="suggestion-button"),
                        html.Button([
                            html.Div(className="suggestion-icon"),
                            html.Span("Aggregate age by gender?")
                        ], id="suggestion-4", className="suggestion-button")
                    ], className="suggestion-buttons")
                ], id="welcome-container", className="welcome-container visible"),
                
                # Chat messages will be added here
                html.Div([],id="chat-messages", className="chat-messages"),
                
            ], id="chat-content", className="chat-content"),
            
            # Fixed chat input at bottom (for after initial message is sent)
            html.Div([
                html.Div([
                    dcc.Input(
                        id="chat-input-fixed",
                        placeholder="Ask your question...",
                        className="chat-input",
                        type="text",
                        disabled=False
                    ),
                    html.Div([
                        html.Button(
                            id="send-button-fixed", 
                            className="input-button send-button",
                            disabled=False
                        )
                    ], className="input-buttons-right"),
                    # Add this tooltip div
                    html.Div("You can only submit one query at a time", 
                            id="query-tooltip", 
                            className="query-tooltip hidden")
                ], id="fixed-input-container", className="fixed-input-container"),
                
                html.Div("Always review the accuracy of responses.", className="disclaimer-fixed")
            ], id="fixed-input-wrapper", className="fixed-input-wrapper"),
            
        ], id="chat-container", className="chat-container"),
        
    ], id="main-content", className="main-content"),
    
    # Add these new components
    html.Div(id='dummy-output'),
    html.Div(id='dummy-output-disable', style={'display': 'none'}),
    dcc.Store(id="chat-trigger", data={"trigger": False, "message": "", "session_id": "", "new_chat_active": True}),
    dcc.Store(id="chat-history-store", data=[]),
    dcc.Store(id="request-cache", data={}),
    dcc.Interval(id='thinking-checker', interval=3000, n_intervals=0),  # Check every 3 seconds
])

# Store chat history
chat_history = []

# Add this function at the beginning of your app.py file, right after the imports

def format_sql_query(sql_query):
    """Format SQL query using sqlparse library"""
    formatted_sql = sqlparse.format(
        sql_query,
        keyword_case='upper',  # Makes keywords uppercase
        identifier_case=None,  # Preserves identifier case
        reindent=True,         # Adds proper indentation
        indent_width=2,        # Indentation width
        strip_comments=False,  # Preserves comments
        comma_first=False      # Commas at the end of line, not beginning
    )
    return formatted_sql



# First callback: Handle inputs and show thinking indicator
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-input-fixed", "value"),
     Output("welcome-container", "className"),
     Output("chat-trigger", "data"),
     Output("chat-list", "children"),
     Output("chat-history-store", "data")],
    [Input("send-button-fixed", "n_clicks"),
     Input("chat-input-fixed", "n_submit"),
     Input("suggestion-1", "n_clicks"),
     Input("suggestion-2", "n_clicks"),
     Input("suggestion-3", "n_clicks"),
     Input("suggestion-4", "n_clicks")],
    [State("chat-input-fixed", "value"),
     State("chat-messages", "children"),
     State("welcome-container", "className"),
     State("chat-list", "children"),
     State("chat-history-store", "data"),
     State("chat-trigger", "data")],
    prevent_initial_call=True,
    throttle=1000  # Throttle to one trigger per second
)
def handle_all_inputs(n_clicks, n_submit, s1, s2, s3, s4, input_value, current_messages, 
                     welcome_class, current_chat_list, chat_history, trigger_data):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle suggestion buttons
    suggestion_messages = {
        "suggestion-1": "What tables are there and how are they connected?",
        "suggestion-2": "Get average age by dose?",
        "suggestion-3": "Explain the dataset",
        "suggestion-4": "Aggregate age by gender?"
    }
    
    # Get the user input based on what triggered the callback
    if trigger_id in suggestion_messages:
        user_input = suggestion_messages[trigger_id]
    else:
        user_input = input_value
    
    if not user_input:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Create user message
    user_message = html.Div([
        html.Div([
            html.Div("S", className="user-avatar"),
            html.Span("Sophie", className="model-name")
        ], className="user-info"),
        html.Div(user_input, className="message-text")
    ], className="user-message message")
    
    # Add the user message to the chat
    if current_messages:
        # If we have existing messages, append to them
        updated_messages = current_messages + [user_message]
    else:
        # If no messages, start new conversation
        updated_messages = [user_message]
    
    # Add thinking indicator
    thinking_indicator = html.Div([
        html.Div([
            html.Span(className="spinner"),
            html.Span("Thinking...")
        ], className="thinking-indicator")
    ], className="bot-message message")
    
    updated_messages.append(thinking_indicator)
    
    is_new_chat = trigger_data.get("new_chat_active", False)

    current_session_id = None
    # Always create a new session if we're in new chat mode
    if is_new_chat:
        # Create a new session ID regardless of existing sessions
        current_session_id = str(uuid.uuid4())
        
        # Create new session entry
        new_session = {
            "session_id": current_session_id,
            "first_query": user_input,
            "queries": [user_input],
            "messages": updated_messages
        }
        
        # Add the new session to the top of the chat history
        chat_history = [new_session] + chat_history
    else:
        # Normal behavior (not new chat mode)
        # Determine if we're adding to an existing chat session or creating a new one
        current_session_id = trigger_data.get("session_id")
        
        # If no current session, create a new one
        if current_session_id is None:
            current_session_id = str(uuid.uuid4())
        
        # Update chat history
        if chat_history is None:
            chat_history = []
        
        # Find if we already have a chat entry for this session
        session_entry = None
        for entry in chat_history:
            if entry.get("session_id") == current_session_id:
                session_entry = entry
                break
        
        if session_entry:
            # Update existing session
            session_entry["messages"] = updated_messages
            session_entry["queries"].append(user_input)
            # Move this session to the top
            chat_history.remove(session_entry)
            chat_history = [session_entry] + chat_history
        else:
            # Create new session entry
            new_session = {
                "session_id": current_session_id,
                "first_query": user_input,
                "queries": [user_input],
                "messages": updated_messages
            }
            chat_history = [new_session] + chat_history
    
    # Update sidebar chat list - always mark the newest session as active
    updated_chat_list = []
    for i, session in enumerate(chat_history):
        class_name = "chat-item active" if i == 0 else "chat-item"
        updated_chat_list.append(
            html.Div(
                session["first_query"],
                className=class_name,
                id={"type": "chat-item", "index": i}
            )
        )
    
    # Set the trigger data with new message
    trigger_data = {
        "trigger": True, 
        "message": user_input, 
        "session_id": current_session_id,
        "new_chat_transition": False,
        "new_chat_active": False  # Reset the new chat flag
    }
    
    return (updated_messages, "", "welcome-container hidden", 
            trigger_data, updated_chat_list, chat_history)


@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True)],
    [Input("chat-trigger", "data")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data")],
    prevent_initial_call=True,
    throttle=1000 
)
def get_model_response(trigger_data, current_messages, chat_history):
    # Add this print statement to debug
    print(f"Model response callback triggered with: {trigger_data}")
    
    if not trigger_data or not trigger_data.get("trigger"):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Use a lock to prevent concurrent executions
        
    try:
        user_input = trigger_data.get("message", "")
        session_id = trigger_data.get("session_id", "")
        
        if not user_input or not session_id:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Pass the session_id to genie_query to maintain conversation context
        response, query_text = genie_query(user_input, session_id)
        #agent_response =  text2sql_agent.invoke(
        #        {"messages": [HumanMessage(content=user_input)],
        #         "session_id":session_id})
            
        if isinstance(response, str) and "Your request is already being processed" in response:
            
            return dash.no_update, dash.no_update, dash.no_update
        
        if isinstance(response, str):
            response = response.replace("`", "")
            content = dcc.Markdown(response, className="message-text")
        else:
            # Data table response
            # Store the dataframe as JSON in a hidden div
            table_id = f"table-{len(chat_history)}"
            
            # Create the table with adjusted styles
            data_table = dash_table.DataTable(
                id=table_id,
                data=response.to_dict('records'),
                columns=[{"name": i, "id": i} for i in response.columns],
                
                # Export configuration
                export_format="csv",
                export_headers="display",
                
                # Other table properties
                page_size=10,
                style_table={
                    'display': 'inline-block',
                    'overflowX': 'auto',
                    'width': '95%',
                    'marginRight': '20px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontSize': '12px',
                    'padding': '4px 10px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont,Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif',
                    'backgroundColor': 'transparent',
                    'maxWidth': 'fit-content',
                    'minWidth': '100px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'borderBottom': '1px solid #eaecef'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
                fill_width=False,
                page_current=0,
                page_action='native'
            )
            formatted_sql = format_sql_query(query_text)
            # Create a container for the toggle and query
            if query_text is not None:
                # Generate a unique index for this query
                query_index = f"{len(chat_history)}-{len(current_messages)}"  # This ensures uniqueness
                
                query_section = html.Div([
                    html.Div([
                        html.Button([
                            html.Span("Show code", id={"type": "toggle-text", "index": query_index})
                        ], 
                        id={"type": "toggle-query", "index": query_index}, 
                        className="toggle-query-button",
                        n_clicks=0)
                    ], className="toggle-query-container"),
                    html.Div([
                        html.Pre([
                            html.Code(formatted_sql, className="sql-code")
                        ], className="sql-pre")
                    ], 
                    id={"type": "query-code", "index": query_index}, 
                    className="query-code-container hidden")
                ], id={"type": "query-section", "index": query_index}, className="query-section")
            
            
            
            # Use the built-in export functionality of DataTable
            content = html.Div([
                html.Div([data_table], style={
                    'marginBottom': '20px',
                    'paddingRight': '5px'
                }),
                query_section,
                html.Div("Click the export button in the table header to download as CSV", 
                         style={"fontSize": "12px", "color": "#666", "marginTop": "-15px", "marginBottom": "10px"})
            ])
        
        # Add the agent here
        #graph_agent_response =  text2sql_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        graph_agent_response =  text2sql_agent.invoke(
               {"messages": [HumanMessage(content=user_input)],"session_id":session_id})
        graph_result_output = graph_agent_response["messages"][-1].content or ""
        # check if the answer includes code. This  regular expression will match code blocks
        # with or without the language specifier (like `python`)
        code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', graph_result_output, re.DOTALL)
        if code_block_match:
            graph_content = dcc.Markdown(graph_result_output, className="message-text")
            code_block = code_block_match.group(1).strip()
            cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
            fig = get_fig_from_code(cleaned_code)
            graph = dcc.Graph(figure=fig)
           
        else:
            graph_content = ""
        

        
        bot_response = html.Div(children=[
        html.Div([
                    html.Div(className="model-avatar"),
                    html.Span("Genie", className="model-name")
                ], className="model-info"),
                html.Div([
                    content,
                    html.Div([
                        html.Div([
                            
                            html.Button(
                                id={"type": "thumbs-up-button", "index": len(chat_history)},
                                className="thumbs-up-button"
                            ),
                            html.Button(
                                id={"type": "thumbs-down-button", "index": len(chat_history)},
                                className="thumbs-down-button"
                            )
                        ], className="message-actions")
                    ], className="message-footer")
                ], className="message-content"),
                html.Div([
                    graph_content,
                    html.Div([
                        html.Div([
                            
                            html.Button(
                                id={"type": "thumbs-up-button", "index": len(chat_history)},
                                className="thumbs-up-button"
                            ),
                            html.Button(
                                id={"type": "thumbs-down-button", "index": len(chat_history)},
                                className="thumbs-down-button"
                            )
                        ], className="message-actions")
                    ], className="message-footer")
                ], className="message-content"),
        ],className="bot-message message")
        
        
        # Find the session containing the thinking indicator based on session_id
        session_with_thinking = None
        session_index = None
        for i, session in enumerate(chat_history):
            if session.get("session_id") == session_id:
                session_with_thinking = session
                session_index = i
                break
        
        if session_with_thinking is not None:
            # Update the correct session's messages - replace thinking indicator with response
            original_messages = session_with_thinking["messages"]
            
            # Check if the last message is a thinking indicator
            if original_messages and "thinking-indicator" in str(original_messages[-1]):
                # Replace thinking indicator with response
                updated_messages = original_messages[:-1] + [bot_response]
                chat_history[session_index]["messages"] = updated_messages
                
                # If this is the active session (index 0), update displayed messages
                if session_index == 0:
                    display_messages = updated_messages
                else:
                    # If it's not the active session, don't update the display
                    display_messages = current_messages
            else:
                # If no thinking indicator found (unusual case), append to messages
                updated_messages = original_messages + [bot_response]
                chat_history[session_index]["messages"] = updated_messages
                
                # Similar logic for display messages
                if session_index == 0:
                    display_messages = updated_messages
                else:
                    display_messages = current_messages
        else:
            # If we can't find the session (unusual case), default to current behavior
            display_messages = current_messages[:-1] + [bot_response]
            
            # Try to handle this gracefully
            if chat_history and len(chat_history) > 0:
                chat_history[0]["messages"] = display_messages
        
        # Reset the trigger data
        reset_trigger_data = {"trigger": False, "message": "", "session_id": session_id, "new_chat_active": False}
        return display_messages, chat_history, reset_trigger_data
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        error_response = html.Div([
            html.Div([
                html.Div(className="model-avatar"),
                html.Span("Genie", className="model-name")
            ], className="model-info"),
            html.Div([
                html.Div(error_msg, className="message-text")
            ], className="message-content")
        ], className="bot-message message")
        
        # Find the session with this session_id
        for i, session in enumerate(chat_history):
            if session.get("session_id") == session_id:
                # Replace thinking indicator with error message
                chat_history[i]["messages"] = session["messages"][:-1] + [error_response]
                
                # Only update display if this is the active session
                if i == 0:
                    display_messages = chat_history[i]["messages"]
                else:
                    display_messages = current_messages
                break
        else:
            # If session not found, update current messages
            display_messages = current_messages[:-1] + [error_response]
        
        return display_messages, chat_history, {"trigger": False, "message": "", "session_id": session_id, "new_chat_active": False}
    
    
# Toggle sidebar and speech button
@app.callback(
    [Output("sidebar", "className"),
     Output("new-chat-button", "style"),
     Output("sidebar-new-chat-button", "style"),
     Output("logo-container", "className"),
     Output("nav-left", "className"),
     Output("left-component", "className"),
     Output("main-content", "className")],
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className"),
     State("left-component", "className"),
     State("main-content", "className")]
)
def toggle_sidebar(n_clicks, current_sidebar_class, current_left_component_class, current_main_content_class):
    if n_clicks:
        if "sidebar-open" in current_sidebar_class:
            # Sidebar is closing
            return "sidebar", {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", "main-content"
        else:
            # Sidebar is opening
            return "sidebar sidebar-open", {"display": "none"}, {"display": "flex"}, "logo-container logo-container-open", "nav-left nav-left-open", "left-component left-component-open", "main-content main-content-shifted"
    # Initial state
    return current_sidebar_class, {"display": "flex"}, {"display": "none"}, "logo-container", "nav-left", "left-component", current_main_content_class

# Add callback for chat item selection
@app.callback(
    [Output("chat-messages", "children", allow_duplicate=True),
     Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True)],
    [Input({"type": "chat-item", "index": ALL}, "n_clicks")],
    [State("chat-history-store", "data"),
     State("chat-list", "children"),
     State("chat-trigger", "data")],
    prevent_initial_call=True
)
def show_chat_history(n_clicks, chat_history, current_chat_list, trigger_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if trigger_data.get("new_chat_active", False):
        # Return the trigger data with new_chat_active still set to True
        return dash.no_update, dash.no_update, dash.no_update, trigger_data
    
    # Get the clicked item index
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_index = json.loads(triggered_id)["index"]
    
    if not chat_history or clicked_index >= len(chat_history):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the session data for this chat
    session_data = chat_history[clicked_index]
    
    # Update active state in chat list
    updated_chat_list = []
    for i, item in enumerate(current_chat_list):
        new_class = "chat-item active" if i == clicked_index else "chat-item"
        updated_chat_list.append(
            html.Div(
                item["props"]["children"],
                className=new_class,
                id={"type": "chat-item", "index": i}
            )
        )
    
    # Clear new_chat_active flag in trigger data to allow normal operation again
    updated_trigger_data = trigger_data.copy()
    updated_trigger_data["new_chat_active"] = False
    
    return session_data["messages"], "welcome-container hidden", updated_chat_list, updated_trigger_data

# Improve the clientside callback to handle both new messages and SQL code toggling
app.clientside_callback(
    """
    function(children, n_clicks) {
        // Scroll chat messages to bottom
        var chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // This will run a bit after the SQL code is shown/hidden
        setTimeout(function() {
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }, 100);
        
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    [Input('chat-messages', 'children'),
     Input({"type": "toggle-query", "index": ALL}, "n_clicks")],
    prevent_initial_call=True
)

# Modify the new chat button callback to create a new session and prevent race conditions
@app.callback(
    [Output("welcome-container", "className", allow_duplicate=True),
     Output("chat-messages", "children", allow_duplicate=True),
     Output("chat-trigger", "data", allow_duplicate=True),
     Output("chat-history-store", "data", allow_duplicate=True),
     Output("chat-list", "children", allow_duplicate=True)],
    [Input("new-chat-button", "n_clicks"),
     Input("sidebar-new-chat-button", "n_clicks")],
    [State("chat-messages", "children"),
     State("chat-trigger", "data"),
     State("chat-history-store", "data"),
     State("chat-list", "children")],
    prevent_initial_call=True
)
def reset_to_welcome(n_clicks1, n_clicks2, chat_messages, chat_trigger, chat_history_store, chat_list):
    # Handle pending query
    if chat_trigger.get("trigger") and chat_history_store:
        # If a query is running, remove the thinking indicator
        session_id = chat_trigger.get("session_id", "")
        for i, session in enumerate(chat_history_store):
            if session.get("session_id") == session_id:
                chat_history_store[i]["messages"] = chat_history_store[i]["messages"][:-1]
                break
    
    # Get the current session ID if there is one
    session_id = chat_trigger.get("session_id", "")
    if session_id:
        # Import the function to clear conversation context
        clear_conversation(session_id)
    
    updated_chat_list = []
    for i, item in enumerate(chat_list):
        updated_chat_list.append(
            html.Div(
                item["props"]["children"],
                className="chat-item",  # Remove active class from all items
                id={"type": "chat-item", "index": i}
            )
        )
    
    # Set flags to indicate we're in a new chat transition
    new_trigger_data = {
        "trigger": False, 
        "message": "", 
        "session_id": "", 
        "new_chat_transition": True,
        "new_chat_active": True  # Add this flag to prevent other callbacks from interfering
    }
    
    # Show welcome screen, clear chat messages
    return "welcome-container visible", [], new_trigger_data, chat_history_store, updated_chat_list



# Add callback to disable input while query is running
@app.callback(
    [Output("chat-input-fixed", "disabled"),
     Output("send-button-fixed", "disabled"),
     Output("new-chat-button", "disabled"),
     Output("sidebar-new-chat-button", "disabled"),
     Output("query-tooltip", "className")],
    [Input("chat-trigger", "data")],
    prevent_initial_call=True
)
def toggle_input_disabled(trigger_data):
    # Show tooltip when query is running, hide it otherwise
    running = trigger_data.get("trigger")
    tooltip_class = "query-tooltip visible" if running else "query-tooltip hidden"
    
    # Disable input and buttons when query is running
    return running, running, running, running, tooltip_class


# Fix the callback for thumbs up/down buttons
@app.callback(
    [Output({"type": "thumbs-up-button", "index": MATCH}, "className"),
     Output({"type": "thumbs-down-button", "index": MATCH}, "className")],
    [Input({"type": "thumbs-up-button", "index": MATCH}, "n_clicks"),
     Input({"type": "thumbs-down-button", "index": MATCH}, "n_clicks")],
    [State({"type": "thumbs-up-button", "index": MATCH}, "className"),
     State({"type": "thumbs-down-button", "index": MATCH}, "className")],
    prevent_initial_call=True
)
def handle_feedback(up_clicks, down_clicks, up_class, down_class):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_type = json.loads(trigger_id)["type"]
    
    if button_type == "thumbs-up-button":
        # Toggle thumbs up, remove thumbs down if active
        new_up_class = "thumbs-up-button active" if "active" not in up_class else "thumbs-up-button"
        new_down_class = "thumbs-down-button"
    else:
        # Toggle thumbs down, remove thumbs up if active
        new_up_class = "thumbs-up-button"
        new_down_class = "thumbs-down-button active" if "active" not in down_class else "thumbs-down-button"
    
    return new_up_class, new_down_class

# Replace the clientside callback with a regular callback for toggling SQL display
@app.callback(
    [Output({"type": "toggle-text", "index": MATCH}, "children"),
     Output({"type": "query-code", "index": MATCH}, "className"),
     Output({"type": "toggle-query", "index": MATCH}, "className")],
    [Input({"type": "toggle-query", "index": MATCH}, "n_clicks")],
    [State({"type": "query-code", "index": MATCH}, "className"),
     State({"type": "toggle-query", "index": MATCH}, "className")],
    prevent_initial_call=True
)
def toggle_sql_code(n_clicks, current_code_class, current_button_class):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Check if the code is currently hidden
    is_hidden = "hidden" in current_code_class
    
    if is_hidden:
        # Show the code
        return "Hide code", "query-code-container", "toggle-query-button active"
    else:
        # Hide the code
        return "Show code", "query-code-container hidden", "toggle-query-button"

# Add this callback to force refresh when thinking indicator gets stuck
@app.callback(
    Output("chat-messages", "children", allow_duplicate=True),
    [Input("thinking-checker", "n_intervals")],
    [State("chat-messages", "children"),
     State("chat-history-store", "data")],
    prevent_initial_call=True
)
def refresh_thinking_indicator(n_intervals, current_messages, chat_history):
    """Check if thinking indicator is stuck and refresh the UI if needed"""
    
    # Skip if no messages
    if not current_messages or not chat_history or len(chat_history) == 0:
        return dash.no_update
    
    # Check if the last message is a thinking indicator
    has_thinking = False
    if current_messages and "thinking-indicator" in str(current_messages[-1]):
        has_thinking = True
    else:
        # No thinking indicator, no need to update
        return dash.no_update
    
    # Check if the active session (at index 0) has a response but UI still shows thinking
    active_session = chat_history[0]
    active_messages = active_session.get("messages", [])
    
    # If active session doesn't have thinking indicator but UI does, force refresh
    if active_messages and "thinking-indicator" not in str(active_messages[-1]):
        print("FORCE REFRESHING: Thinking indicator stuck, updating UI with latest messages")
        return active_messages
    
    return dash.no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True,host = '127.0.0.1',port=8050)



@app.callback(
    Output("login-popup", "is_open"),
    Output("login-error", "children"),
    Input("llm-dropdown", "value"),
    prevent_initial_call=True
)
def handle_llm_selection(selected_llm):
    if not selected_llm:
        return True, ""
    allowed_llms = fetch_allowed_llms()
    if selected_llm in allowed_llms:
        return False, ""
    else:
        return True, f"Access denied: {selected_llm} is not permitted."
