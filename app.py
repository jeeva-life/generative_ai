import streamlit as st
from itertools import zip_longest
import os
from dotenv import load_dotenv

from langchain_community import chat_models
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

import emoji

load_dotenv()

OPENAI_API_KEY =os.environ.get("OPENAI_API_KEY") #fetches the value of the OPENAI_API_KEY from the environment variables
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
serpapi = os.environ.get("SERPAPI_API_KEY")

#This function creates a text input field and a "Send" button in the Streamlit sidebar. 
# #When the button is clicked, it returns the entered text; otherwise, it returns None.
def get_text():
    input_text = st.sidebar.text_input("Input: ", key='input')
    if st.sidebar.button('Send'):
        return input_text
    return None

#function processes a list of messages, appending user inputs as "input: ..." and assistant 
#responses as "output: ..." into a single string. It returns the combined conversation history as a formatted string.
def get_history(history_list):
    history=''
    for message in history_list:
        if message['role'] == 'user':
            history += 'input: ' + message['content']
        elif message['role'] == 'assistant':
            history += 'output: ' + message['content']

    return history


def get_response(history, user_message, temperature=0):
    print('sending input to chatgpt')
    DEFAULT_TEMPLATE = """
    As an AI-driven digital journalist, you specialize in understanding, summarizing, 
    and presenting information from trusted news sources. You stay updated on current 
    trends and popular news topics, offering users accurate, verified, and impartial insights 
    in a conversational manner. Users engage with you to explore the latest headlines and stay 
    informed about topics and stories of interest. Your priority in every interaction is to deliver 
    clear, timely, and precise information, building upon the context of previous conversations.
    
    Relevant pieces of previous conversation:  
    {context},  

    Useful news information from Web:  
    {web_knowledge},  

    Current conversation:  
    Human: {input}  
    News Journalist:"""

    #creates a PromptTemplate object with placeholders for context, web_knowledge, and input, using the string 
    # defined in DEFAULT_TEMPLATE. It formats the template dynamically by substituting these placeholders with actual values during runtime.
    PROMPT = PromptTemplate(input_variables=['context', 'web_knowledge', 'input'], template=DEFAULT_TEMPLATE)

    chat_gpt = ChatOpenAI(temperature=temperature, model_name='gpt-3.5-turbo')

    tools = load_tools(['serpapi'], llm=chat_gpt)

    #It initializes an AI agent using specified tools and the chat_gpt model, with a "zero-shot react" reasoning 
    #strategy for task execution. It also enables verbose logging and specifies error handling to verify output correctness.
    agent = initialize_agent(tools, chat_gpt, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                            handle_parsing_errors='Check output and make sure it confirms')

    # It uses the agent to execute a task where it fetches detailed, unsummarized analysis from news articles related to 
    # the user_message. It runs the query dynamically, leveraging the agent's tools and reasoning to retrieve relevant information.
    web_knowledge = agent.run("fetch detailed analysis without summarizing from news articles regarfing " + user_message)

    #It creates an LLMChain object to define a workflow where the chat_gpt model interacts with a specified PROMPT. 
    # Setting verbose=True ensures detailed logging of the chain's operations for better debugging and transparency.
    conversation = LLMChain(
        llm=chat_gpt,
        prompt=PROMPT,
        verbose=True
    )
    
    response=conversation.predict(context=history, input=user_message, web_knowledge=web_knowledge)
    return response


st.title("News Collector")

if "past" not in st.session_state:
    st.session_state["past"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = []


user_input = get_text()

if user_input:
    user_history = list(st.session_state['past'])
    bot_history = list(st.session_state['generated'])

    combined_history = []

    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': 'user_msg'})
        if bot_msg is not None:
           combined_history.append({'role': 'assistant', 'content': 'bot_msg'})

    formatted_history = get_history(combined_history)

    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            st.markdown(emoji.emojize(f":speech_balloon: **User**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant**: {st.session_state['generated'][i]}"))
