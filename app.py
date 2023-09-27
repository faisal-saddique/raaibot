from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAAI - Chatbot", page_icon="🎭")

def get_starting_questions():
    output_parser = CommaSeparatedListOutputParser()

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="You are an AI language model assistant. Your task is to generate five different questions that a user can ask from a {subject}. Don't add numbers in the start.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = OpenAI(temperature=0)

    _input = prompt.format(subject="relationship expert")
    output = model(_input)

    res = output_parser.parse(output)
    return res

def set_starting_question(question):
    st.session_state["starting_question"] = question

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.title("🎭 RAAI - Chatbot")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

template = """You are a caring relationship expert representing RAAI. Your main mission is to assist individuals in navigating their personal relationships by providing empathetic guidance. Your strength lies in fostering warm and understanding conversations. You listen attentively to people's relationship concerns and ask for more details when necessary. Your communication style is intentionally designed to be relatable and human-like, allowing you to incorporate emojis naturally throughout the conversation, just as a real person would. This creates a seamless and friendly interaction that feels genuine and approachable.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=ChatOpenAI(streaming=True,temperature=.7), prompt=prompt, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if "display_starting_question" not in st.session_state:
    st.session_state["display_starting_question"] = True

if "starting_question" not in st.session_state:
    st.session_state["starting_question"] = None



if st.session_state["display_starting_question"]:
    res = get_starting_questions()
    # Create two columns
    col1, col2 = st.columns(2)

    # Display the first three questions in the first column
    with col1:
        for question in res[:3]:
            if st.button(question, key=question, use_container_width=True, on_click=lambda q=question: set_starting_question(q)):
                starting_question = question

    # Display the next two questions in the second column
    with col2:
        for question in res[3:]:
            if st.button(question, key=question, use_container_width=True, on_click=lambda q=question: set_starting_question(q)):
                starting_question = question

    st.session_state["display_starting_question"] = False

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input() or st.session_state["starting_question"]:
    
    if st.session_state["starting_question"]:
        st.chat_message("human").write(st.session_state["starting_question"])
        st.session_state["starting_question"] = None
    else:
        st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        response = llm_chain.run(prompt,callbacks=[stream_handler])