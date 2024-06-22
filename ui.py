import streamlit as st
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.ai import AIMessageChunk
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.llms import HuggingFaceTextGenInference
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import gc


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

@st.cache_resource
def create_gclient():
	creds = ServiceAccountCredentials.from_json_keyfile_name('finetuning-storage-beba143e31e5.json', scope)
	client = gspread.authorize(creds)
	return client

sheets_connection = create_gclient()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
	inference_server_url = st.secrets["inference_url"]
	llm = HuggingFaceTextGenInference(
		inference_server_url=inference_server_url,
		max_new_tokens=800,
		temperature=1.35,
		repetition_penalty=1.2,
		top_k=10,
		typical_p=0.95,
		server_kwargs={
		"headers": {
			"Authorization": f"""Bearer {st.secrets["HF_TOKEN"]}""",
			"Content-Type": "application/json",
		}
		}
	)
	st.session_state.llm = llm

if "curr_response" not in st.session_state:
	st.session_state.curr_response = ""

if "feedback" not in st.session_state:
	st.session_state.feedback = False

if "chat_model" not in st.session_state:
	st.session_state.chat_model = ChatHuggingFace(llm=st.session_state.llm , model_id="Neohumans-ai/Eli-Hindi-v0.1")

if "langchain_messages" not in st.session_state:
	st.session_state.langchain_messages = []


def start_callback():
	gc.collect()
	if st.session_state.feedback == True:
		st.error("Finalise the previous response to proceed!!!!")
		st.sleep(10)

	if st.session_state.system_prompt == None or st.session_state.system_prompt == "":
		st.error("Enter System Prompt To proceed.")
		st.stop()
	
	st.session_state.curr_response = ""
	st.session_state.feedback = True	
	if len(st.session_state.messages) == 10:
		message_to_be_saved = st.session_state.messages.copy()
		message_to_be_saved.insert(0,{"role":"system","content":st.session_state.system_prompt})
		data = [st.experimental_user.email,json.dumps(message_to_be_saved)]
		sh = sheets_connection.open('RLHF_DATA').worksheet('data')
		sh.append_row(data)
		st.session_state.messages = []
		st.session_state.langchain_messages = []

	if len(st.session_state.messages) == 0:
		st.session_state.langchain_messages.append(SystemMessage(content=st.session_state.system_prompt))


def clear_callback():
	if len(st.session_state.messages) > 0:
		message_to_be_saved = st.session_state.messages.copy()
		message_to_be_saved.insert(0,{"role":"system","content":st.session_state.system_prompt})
		data = [st.experimental_user.email,json.dumps(message_to_be_saved)]
		sh = sheets_connection.open('RLHF_DATA').worksheet('data')
		sh.append_row(data)
		st.session_state.messages = []
		st.session_state.langchain_messages = []
		st.session_state.feedback = False


with st.sidebar:
	system_prompt = st.text_area("System Prompt",key="system_prompt")

st.title("CHAT PREFERENCE UI")

button = st.button("Clear Chat",on_click=clear_callback)


for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])


if user_input := st.chat_input("Talk in hinglish",on_submit=start_callback):
	with st.chat_message("user"):
		st.markdown(user_input.strip())

	st.session_state.langchain_messages.append(HumanMessage(content=user_input.strip()))
	st.session_state.messages.append({"role":"user","content":user_input.strip()})

	with st.spinner("Loading"):
		response = st.session_state.chat_model.invoke(st.session_state.langchain_messages)
		st.session_state.curr_response = response.content

if st.session_state.feedback == True and st.session_state.curr_response!="":
	with st.form("Edit form",clear_on_submit=True):
		assistant_final_response = st.text_area("Finalise the response",value=st.session_state.curr_response)
		submitted = st.form_submit_button("Save the response")
		if submitted:
			if len(st.session_state.messages) > 0 and (st.session_state.curr_response != assistant_final_response):
				message_to_be_saved1 = st.session_state.messages.copy()
				message_to_be_saved2 = st.session_state.messages.copy()
				
				message_to_be_saved1.insert(0,{"role":"system","content":st.session_state.system_prompt})
				message_to_be_saved2.insert(0,{"role":"system","content":st.session_state.system_prompt})				

				message_to_be_saved1.append({"role":"assistant","content":st.session_state.curr_response.strip()})
				message_to_be_saved2.append({"role":"assistant","content":assistant_final_response.strip()})
				
				data = [st.experimental_user.email,json.dumps(message_to_be_saved1),json.dumps(message_to_be_saved2)]
				sh = sheets_connection.open('RLHF_DATA').worksheet('dpo')
				sh.append_row(data)
				
			st.session_state.langchain_messages.append(AIMessageChunk(content=assistant_final_response.strip()))
			st.session_state.messages.append({"role":"assistant","content":assistant_final_response.strip()})
			st.session_state.feedback = False
			st.rerun()
		

















