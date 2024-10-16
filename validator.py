import streamlit as st
import random
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import ast
import openai 
import numpy as np
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# Access the environment variables
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")

# Integrating OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(question, steps=None):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant capable of answering a wide variety of fact-based questions. Provide direct and accurate answers across different topics, always ensuring precision and clarity."
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        if steps:
            messages[1]["content"] = f"The user provided the following steps to solve the question:\n{steps}\n\nThe original question is: {question}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5,
            # temperature=0.2 if steps else 0.5,
            max_tokens=1000 if steps else 2048,
            top_p=1,
            frequency_penalty=0.5 if steps else 0,
            presence_penalty=0.5 if steps else 0
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error fetching response: {str(e)}"

def validate_input(input_value):
    return bool(input_value)

# Function to count tokens using tiktoken
def count_tokens(text, model_name='gpt-4'):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

# Connecting Database on GCP
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    
    engine = conn
except Exception as e:
    engine = None
    st.error(f"Database connection error: {str(e)}")

# Queries
if engine:
    try:
        query = 'SELECT * FROM gaia.data;'
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.sidebar.error(f"Error executing query: {str(e)}")
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

# Data Loading Functions
def get_or_create_task(level, prompt, actual_answer):
    try:
        cursor.execute("SELECT req_id FROM gaia.tasks WHERE prompt = %s", (prompt,))
        existing_task = cursor.fetchone()
        
        if existing_task:
            return existing_task[0]
        else:
            cursor.execute("""
                INSERT INTO gaia.tasks (level, prompt, actual_answer)
                VALUES (%s, %s, %s)
                RETURNING req_id
            """, (level, prompt, actual_answer))
            new_req_id = cursor.fetchone()[0]
            conn.commit()
            return new_req_id
    except Exception as e:
        st.sidebar.error(f"Error in get_or_create_task: {e}")
        conn.rollback()
        return None

def insert_execution(req_id, input_token, output_token, total_tokens):
    try:
        cursor.execute("""
            INSERT INTO gaia.executions (req_id, input_token, output_token, total_tokens) VALUES (%s, %s, %s, %s) RETURNING execution_id
        """, (req_id, input_token, output_token, total_tokens))
        execution_id = cursor.fetchone()[0]
        conn.commit()
        return execution_id
    except Exception as e:
        st.sidebar.error(f"Error inserting into executions table: {e}")
        conn.rollback()
        return None
    
def insert_step_run(execution_id, steps, generated_answer, input_token, output_token, total_tokens):
    try:
        cursor.execute("""
            INSERT INTO gaia.stepruns (execution_id, steps, generated_answer, isMatch, input_token, output_token, total_tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING step_run_id
        """, (execution_id, steps, generated_answer, False, input_token, output_token, total_tokens))
        step_run_id = cursor.fetchone()[0]
        conn.commit()
        return step_run_id
    except Exception as e:
        st.sidebar.error(f"Error inserting into stepruns table: {e}")
        conn.rollback()
        return None

def update_step_run_match(step_run_id, is_match):
    try:
        cursor.execute("""
            UPDATE gaia.stepruns
            SET isMatch = %s
            WHERE step_run_id = %s
        """, (is_match, step_run_id))
        conn.commit()
        return True
    except Exception as e:
        st.sidebar.error(f"Error updating stepruns table: {e}")
        conn.rollback()
        return False

# User Interface
st.header("Validator Tool")

st.write('')

# Sidebar Navigation
levels = ['All'] + sorted(df['level'].unique().tolist()) if not df.empty else ['All']
selected_level = st.sidebar.selectbox("Select difficulty level:", levels)

## Select Level
if selected_level != 'All' and not df.empty:
    filtered_df = df[df['level'] == selected_level]
    questions = filtered_df['question'].tolist()
else:
    questions = df['question'].tolist() if not df.empty else []

if 'dropdown_value' not in st.session_state:
    st.session_state.dropdown_value = questions[0] if questions else None
if 'openai_response' not in st.session_state:
    st.session_state.openai_response = "Run Prompt to get an answer from ChatGPT"
if 'execution_id' not in st.session_state:
    st.session_state.execution_id = None
if 'step_run_id' not in st.session_state:
    st.session_state.step_run_id = None

## Select Prompt
dropdown_value = st.sidebar.selectbox(
    "Choose a prompt to test", 
    questions, 
    index=questions.index(st.session_state.dropdown_value) if st.session_state.dropdown_value in questions else 0,
    key="question_dropdown"
)

if dropdown_value != st.session_state.dropdown_value:
    st.session_state.dropdown_value = dropdown_value
    st.session_state.openai_response = "Run Prompt to get an answer from ChatGPT"
    st.session_state.execution_id = None
    st.session_state.step_run_id = None

## Randomly Choose Prompts
if st.sidebar.button("Randomize", key="randomize_button"):
    if questions:
        st.session_state.dropdown_value = random.choice(questions)
        st.session_state.openai_response = "Run Prompt to get an answer from ChatGPT"
        st.session_state.execution_id = None
        st.session_state.step_run_id = None
        st.rerun()
    else:
        st.sidebar.error("No questions available for the selected level.")

## Prompt Text
st.sidebar.header("Prompt:")
prompt_text = st.session_state.dropdown_value if st.session_state.dropdown_value else "No question selected"
st.sidebar.write(prompt_text)

## Run Prompt Button
if st.sidebar.button("Run Prompt", key="run_prompt_button"):
    if validate_input(st.session_state.dropdown_value):
        st.session_state.openai_response = "Fetching response from ChatGPT..."
        
        input_tokens_count = count_tokens(st.session_state.dropdown_value)
        
        openai_response = get_openai_response(st.session_state.dropdown_value)
        
        output_tokens_count = count_tokens(openai_response)
        
        total_tokens = input_tokens_count + output_tokens_count
        
        st.sidebar.write(f"Total Input Tokens: {input_tokens_count}")
        st.sidebar.write(f"Total Output Tokens: {output_tokens_count}")
        st.sidebar.write(f"Total Tokens Used: {total_tokens}")
        
        st.session_state.openai_response = openai_response
        
        selected_question = st.session_state.dropdown_value
        
        question_data = df[df['question'] == selected_question].iloc[0]
        
        level = int(question_data['level']) if isinstance(question_data['level'], np.integer) else question_data['level']
        
        actual_answer = question_data['final_answer']
        
        req_id = get_or_create_task(level, selected_question, actual_answer)
        
        if req_id is not None:
            execution_id = insert_execution(req_id, input_tokens_count, output_tokens_count, total_tokens)
            if execution_id:
                st.session_state.execution_id = execution_id
                st.sidebar.success(f"Execution recorded with ID: {execution_id}")
            else:
                st.sidebar.error("Failed to record execution")     
    else:
        st.sidebar.error("Please select a valid question.")

col1, col2 = st.columns(2)

question_answer_dict = dict(zip(df['question'], df['final_answer'])) if not df.empty else {}
final_answer = question_answer_dict.get(st.session_state.dropdown_value, "No final answer found.")

# Main Section
with col1:
    st.markdown("##### Actual Answer")
    st.write(final_answer)

with col2:
    st.markdown("##### ChatGPT Answer")
    st.write(st.session_state.openai_response)

st.divider()

question_steps_dict = dict(zip(df['question'], df['annotator_metadata'])) if not df.empty else {}
annotator_data_str = question_steps_dict.get(st.session_state.dropdown_value, "{}")
try:
    annotator_data = ast.literal_eval(annotator_data_str)
except ValueError:
    annotator_data = {}

## Button for storing matched answers
if st.button("Answers Match", key="answers_match_button"):
    if st.session_state.step_run_id:
        if update_step_run_match(st.session_state.step_run_id, True):
            st.success("Answers marked as matching!")
        else:
            st.error("Failed to update match status.")
    else:
        st.error("No active step run. Please re-run the prompt first.")

steps = annotator_data.get('Steps', "No steps found.")

z = (dropdown_value or "No value selected") + '\n' + '\n' + steps

st.markdown("##### Steps followed:")
steps = st.text_area("Edit these steps and run again if validation fails", z, height=300)


## Button for running prompts again
if st.button("Re-run Prompt", key="re_run_prompt_button"):
    if validate_input(st.session_state.dropdown_value) and validate_input(steps):
        st.session_state.openai_response = "Fetching response from ChatGPT..."

        re_input_tokens_count = count_tokens(steps)
        
        re_run_response = get_openai_response(steps)

        re_output_tokens_count = count_tokens(re_run_response)
        
        re_total_tokens = re_input_tokens_count + re_output_tokens_count

        st.session_state.re_run_input_tokens = re_input_tokens_count
        st.session_state.re_run_output_tokens = re_output_tokens_count
        st.session_state.re_run_total_tokens = re_total_tokens
        
        st.session_state.openai_response = re_run_response

        if st.session_state.execution_id:
            step_run_id = insert_step_run(st.session_state.execution_id, steps, re_run_response, re_input_tokens_count, re_output_tokens_count, re_total_tokens)
            if step_run_id:
                st.session_state.step_run_id = step_run_id
                st.sidebar.success(f"Step run recorded with ID: {step_run_id}")
            else:
                st.sidebar.error("Failed to record step run")
        else:
            st.sidebar.error("No active execution. Please run the prompt first.")

        st.rerun()
    else:
        st.error("Please provide both a valid question and steps to re-run the prompt.")
        st.session_state.openai_response = "Run Prompt to get an answer from ChatGPT"

if 're_run_input_tokens' in st.session_state:
    st.write(f"Re-run Input Tokens: {st.session_state.re_run_input_tokens}")
if 're_run_output_tokens' in st.session_state:
    st.write(f"Re-run Output Tokens: {st.session_state.re_run_output_tokens}")
if 're_run_total_tokens' in st.session_state:
    st.write(f"Total Re-run Tokens Used: {st.session_state.re_run_total_tokens}")