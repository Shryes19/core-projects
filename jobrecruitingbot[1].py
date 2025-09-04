import llama_index
from llama_index.core import Settings,VectorStoreIndex,Document,SimpleDirectoryReader,load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import os
import pandas
import json
from prompts import instruction_str,new_prompt,context
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import streamlit as st
def get_api_key(file_name):
    with open(file_name, 'r') as file:
        return file.read().strip()
llm=Gemini(api_key=get_api_key("key.txt"),
           model_name="models/gemini-2.0-flash")
Settings.llm=llm
profile_path=r"/Users/shrye/Downloads/archive/LinkedIn people profiles datasets.csv"
profile_df=pandas.read_csv(profile_path)
profile_query_engine=PandasQueryEngine(llm=llm, df=profile_df, verbose=True,
                                       instruction_str=instruction_str)
profile_query_engine.update_prompts({"pandas_prompt":new_prompt})
#from pdf import physics_engine
tools=[
    QueryEngineTool(query_engine=profile_query_engine,
                    metadata=ToolMetadata(name="LinkedIn_profiles_data",
                                          description="This gives information about the profiles of everyone "
                                                      "in LinkedIn based on their city"),
                    ),
    # QueryEngineTool(query_engine=physics_engine,
    #                 metadata=ToolMetadata(name="physics",
    #                                       description="This gives detailed information about electric charges "),
    #                 ),

]


agent=ReActAgent.from_tools(tools,llm=llm,verbose=True,context=context)

# st.title("Job Profile Bot")

# prompt=st.chat_input("Enter your requirements")

# if prompt:  
#     # try:  
#     #     result = agent.query(prompt)  # Call the agent.query with valid prompt  
#     #     # st.write("Query Result:", result)  # Display the results 
#     #     st.markdown(result)
#     #     print(result) 
#     # except Exception as e:  
#     #     st.error(f"An error occurred while processing your query: {str(e)}") 
#     # st.markdown(f"**Your Question:** {prompt}")  # Display the user's prompt  
#     try:  
#         # Display the user prompt  
#         st.markdown(f"**Your Question:** {prompt}")  # Display the user's prompt  
#         result = agent.query(prompt)  # Call the agent.query with valid prompt  
        
#         # Display only the answer  
#         if isinstance(result, (list, dict)):  
#             formatted_result = json.dumps(result, indent=2)  # Pretty print for JSON-like structures  
#             st.text(formatted_result)  # Display formatted text  
#         else:  
#             st.text(result)  # Display single string result  
            
#     except Exception as e:  
#         st.error(f"An error occurred while processing your query: {str(e)}")   
# else:  
#     # st.warning("Please enter a query to ask Gemini.") 
#     pass

# def display_profile_info(profile_info):  
#     for key, value in profile_info.items():  
#         st.write(f"**{key.replace('_', ' ').capitalize()}:** {value}")  

# if prompt:  
#     try:  
#         st.markdown(f"**Your Question:** {prompt}")  
#         result = agent.query(prompt)  
        
#         if isinstance(result, str):  
#             st.write(result)  
#         elif isinstance(result, (list, dict)):  
#             display_profile_info(result)  
#         else:  
#             st.write(str(result))  
#     except Exception as e:  
#         st.error(f"An error occurred while processing your query: {str(e)}")
# 
st.markdown("""  
    <style>  
        body {  
            background-image: linear-gradient(to right, rgba(255,0,0,0), rgba(255,0,0,1));  
            font-family: 'Arial', sans-serif;  
        }  
        [data-testid="stAppViewContainer"]{
            background-image: linear-gradient(to left, black, black , blue); 
        }  
        .title {  
            color: blue;  
            text-align: center;  
            margin: 20px 0;  
            font-size: 4em;  
            position: relative;
            top: 10%;
        }  
        .description {  
            font-size: 1.2em;  
            text-align: center;  
            margin-bottom: 20px;  
            color: white;  
        }        
        .input-container {   
            margin-bottom: 20px;  
        }  
        # .input-box {  
        #     width: 60%;  
        #     padding: 10px;  
        #     font-size: 1em;  
        #     border: 1px solid blue;  
        #     border-radius: 5px;  
        # }  
        [class="st-ae st-af st-ag st-ah st-ai st-aj st-ak st-al st-am st-an st-ao st-ap st-aq st-ar st-as st-at st-au st-av st-aw st-ax st-ay st-az st-b0 st-b1 st-b2 st-b3 st-b4 st-b5 st-b6 st-b7 st-b8 st-b9 st-ba"]: focus{
            width: 60%;  
            padding: 10px;  
            font-size: 1em;  
            border: 1px solid blue;  
            border-radius: 2px;  
        }    
        .output {    
            border-radius: 5px;
            background-color: grey;  
            padding: 15px;  
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);  
            margin-top: 20px;  
        }  
        .error {  
            color: blue;  
        }
        .image{
            width: 900px;
            height: 100px;    
            margin: 0px;
        }  
        .stTextInput input:focus {  
        border: 2px solid blue ;
        border-radius: 15px;      
        box-shadow: 0 0 5px blue ; 
        outline: none;     
        }  
        .stTextInput input {  
            border: 2px solid blue ; 
            border-radius: 15px; 
        }     
        .question {  
            font-weight: bold;  
            margin-bottom: 10px;  
        }  
        .stAppHeader st-emotion-cache-12fmjuu ezrtsby2{
          background-color: black;    
        }    
        .stMainBlockContainer block-container st-emotion-cache-13ln4jf ea3mdgi5{
            border: 2px solid black;
            }    
    </style>  
""", unsafe_allow_html=True)  

# Streamlit UI 
st.markdown('<img src="https://www.logo.wine/a/logo/LinkedIn/LinkedIn-Logo.wine.svg" class="image"/>',unsafe_allow_html=True)
st.markdown('<div class="title">Job Profile Bot</div>', unsafe_allow_html=True)  
st.markdown('<div class="description">Get information from LinkedIn profiles based on specific queries!</div>', unsafe_allow_html=True)  
st.sidebar.header('chat history')
# User input for queries  
prompt = st.text_input("Get employee information:", "", key="input", placeholder="Enter your requirements")  

def display_profile_info(profile_info):  
    for key, value in profile_info.items():  
        st.write(f"**{key.replace('_', ' ').capitalize()}:** {value}")  
# if st.session_state.query_history:  
#     st.markdown("### Query History")  
#     for history in st.session_state.query_history:  
#         st.write(f"**Query:** {history['query']}  \n**Result:** {history['result']}")  

if "history" not in st.session_state:
    st.session_state.history = []
if prompt:  
    try:  
        st.markdown(f'<div class="question">Your Question:</div> <div class="output">{prompt}</div>', unsafe_allow_html=True)  
        result = agent.query(prompt)
        st.session_state.history.append({"prompt": prompt, "result": result}) 
        if isinstance(result, str):  
            st.write(f'<div class="output">{result}</div>', unsafe_allow_html=True)

        elif isinstance(result, (list, dict)):  
            display_profile_info(result) 

            # st.session_state.query_history.append({'query': prompt, 'result': result}) 
        else:  
            st.write(f'<div class="output">{str(result)}</div>', unsafe_allow_html=True)     
    except Exception as e:  
        st.error(f'<div class="error">An error occurred while processing your query: {str(e)}</div>', unsafe_allow_html=True)  
for entry in st.session_state.history:
    st.sidebar.write(f"**Question:** {entry['prompt']}")
    if isinstance(entry["result"], str):
        st.sidebar.write(f"**Answer:** {entry['result']}")
    elif isinstance(entry["result"], (list, dict)):
        st.sidebar.write("**Answer:**")
        st.sidebar.json(entry["result"])  
    else:
        st.sidebar.write(f"**Answer:** {str(entry['result'])}")
   


