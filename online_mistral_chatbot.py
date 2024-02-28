from langchain_community.llms import HuggingFaceEndpoint
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
from deep_translator import GoogleTranslator
import streamlit as st

system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
st.markdown(system_prompt)

@st.cache_resource
def create_chain():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=5000,
        temperature=0.5,
        token=HUGGINGFACEHUB_API_TOKEN
    )
    model = Llama2Chat(llm=llm)
    template_messages = [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt = ChatPromptTemplate.from_messages(template_messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_chain = LLMChain(llm=model, prompt=prompt, memory=memory)
    
    st.markdown("Model Loaded Successfully....")
    
    return llm_chain


# Set the webpage title
st.set_page_config(
    page_title="Alaa's Chat Robot!"
)

# Create a header element
st.header("Alaa's Chat Robot!")


# Create Select Box
lang_opts = ["ar", "en" , "fr" , "zh-CN"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)


# Create LLM chain to use for our chatbot.
mod = create_chain()


# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def disable():
    st.session_state.disabled = True
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False


async def get_response(mod , user_prompt , container):
    full_response = ""
    # Add the response to the chat window
    with container.chat_message("assistant"):
        full_response = mod.run(user_prompt)
        container.markdown(full_response)
    full_response = GoogleTranslator(source='auto', target=lang_selected).translate(full_response)
    #container.markdown(full_response)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input" , on_submit = disable , disabled=st.session_state.disabled):
    del st.session_state.disabled
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
    #st.chat_input("Your message here", key="disabled_chat_input", disabled=True)
    st.markdown("in session")
    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

        
    user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
    asyncio.run(get_response(mod , user_prompt , st.empty()))
        
##
        
    st.rerun()
