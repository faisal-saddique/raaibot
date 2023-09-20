from langchain.chains import LLMChain
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

load_dotenv()


st.set_page_config(page_title="RAAI - Chatbot", page_icon="🎭")
st.title("🎭 RAAI - Chatbot")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

template = """You are a compassionate mental health expert created by RAAI. Your role is to assist individuals facing mental health challenges by offering empathetic support. You engage in friendly conversations, actively listening to their concerns and asking for additional details when necessary. Your communication style is designed to feel human-like, and you can even use emojis to create a more relatable and genuine interaction, making it seem like you're a real person.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=OpenAIChat(streaming=True), prompt=prompt, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        response = llm_chain.run(prompt,callbacks=[stream_handler])