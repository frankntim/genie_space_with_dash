
import pandas as pd
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from lifelines import KaplanMeierFitter, NelsonAalenFitter, BreslowFlemingHarringtonFitter, CoxPHFitter,  WeibullFitter
#from langchain.chat_models import ChatOpenAI
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from databricks.sdk import WorkspaceClient
#from databricks_langchain import (
#    ChatDatabricks,
#    UCFunctionToolkit,
#)
# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.preprocessing import LabelEncoder
import re
import plotly.tools as tls
import plotly.io as pio

CHART_PROMPT = '''
    Role: You are biomarker assistant tasked with generating Cox Fitting and providing technical analysis and explanation of Cox Proportional Hazard Models.

    Objective: Your job is to explain Cox Fitting summary, provide publication ready visual plots.

    Capabilities: You have been given a dataframe as data. The data columns are provided. The Cox Fitting summary is also provided.

    Next Steps: Explain the already fitted results summary, provide publication ready visual plots, with p-values annotated


    columns: {columns}

    data: {data}

    summary: {summary}

    prompt: {task}

    '''


chart_prompt = ChatPromptTemplate.from_template(CHART_PROMPT)
graph_agent_chain = chart_prompt | LLM


cph = CoxPHFitter()
df = pd.read_csv('survival_rossi.csv')
cph.fit(df, duration_col="week_duration", event_col="arrest_status")
summary_df = cph.summary.reset_index().round(3)


graph_agent_response = graph_agent_chain.invoke({
            "columns":df.columns.tolist(),
            "data": df.to_dict(orient="records"),
            "summary": summary_df.to_dict(orient="records"),
            "task": "provide technical explanation of the Cox Fitting summary and please generate a provide publication ready visual plots of hazard ratios with confidence intervals, with p-values annotated"
        })