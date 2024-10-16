import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sqlalchemy import create_engine
from validator import df
from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv()

# Access the environment variables
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")

# DB Connections
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
        query = 'SELECT * FROM gaia.tasks'
        df_tasks = pd.read_sql(query, conn)
    except Exception as e:
        st.sidebar.error(f"Error executing query: {str(e)}")
        df_tasks = pd.DataFrame()
else:
    df_tasks = pd.DataFrame()

if engine:
    try:
        query = 'SELECT * FROM gaia.executions'
        df_executions = pd.read_sql(query, conn)
    except Exception as e:
        st.sidebar.error(f"Error executing query: {str(e)}")
        df_executions = pd.DataFrame()
else:
    df_executions = pd.DataFrame()

if engine:
    try:
        query = 'SELECT * FROM gaia.stepruns'
        df_stepruns = pd.read_sql(query, conn)
    except Exception as e:
        st.sidebar.error(f"Error executing query: {str(e)}")
        df_stepruns = pd.DataFrame()
else:
    df_stepruns = pd.DataFrame()


# Token Metrics
total_prompts = df['task_id'].nunique()
prompts_tested = df_tasks['req_id'].nunique()

input_tokens_run =  df_executions['input_token'].sum()
input_tokens_rerun =  df_stepruns['input_token'].sum()

output_tokens_run =  df_executions['output_token'].sum()
output_tokens_rerun =  df_stepruns['output_token'].sum()

total_input_tokens = input_tokens_run+input_tokens_rerun
total_input_tokens_cost = (total_input_tokens*0.000005)
total_output_tokens = output_tokens_run+output_tokens_rerun
total_output_tokens_cost = (total_output_tokens*0.000015)
total_cost = total_input_tokens_cost+total_output_tokens_cost

# Sidebar Navigation
st.sidebar.markdown(f"""
#### *Total Prompts*
<p style='font-size: 40px; font-weight: bold; line-height: 2rem;'>{total_prompts}</p>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
#### *Total Prompts Tested*
<p style='font-size: 40px; font-weight: bold; line-height: 2rem;'>{prompts_tested}</p>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
#### *Total Input Tokens Used*
<p style='font-size: 40px; font-weight: bold; line-height: 2rem;'>{total_input_tokens}</p>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
#### *Total Output Tokens Used*
<p style='font-size: 40px; font-weight: bold; line-height: 2rem;'>{total_output_tokens}</p>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
#### *Total Cost*
<p style='font-size: 40px; font-weight: bold; line-height: 2rem;'>${round(total_cost, 4)}</p>
""", unsafe_allow_html=True)

# Main Section

## Steps Frequency Line Chart & Pie Chart for Matched Answers vs Non-Matched Answers
if not df_tasks.empty and not df_executions.empty and not df_stepruns.empty:
    st.header("Step Run Frequency Across Different Prompts")
    
    levels = ['All'] + sorted(df_tasks['level'].unique().tolist())
    selected_level = st.selectbox("Select difficulty level:", levels)

    if selected_level != 'All':
        filtered_executions = df_executions[df_executions['req_id'].isin(
            df_tasks[df_tasks['level'] == selected_level]['req_id']
        )]
    else:
        filtered_executions = df_executions

    filtered_df = df_stepruns[df_stepruns['execution_id'].isin(filtered_executions['execution_id'])]

    execution_counts = filtered_df.groupby('execution_id').size().reset_index(name='step_runs')

    col1, col2 = st.columns(2)

    with col1:
        fig_line = px.line(execution_counts, x='execution_id', y='step_runs', title=f'Executions vs Step Runs for Level {selected_level}')
        st.plotly_chart(fig_line)

    with col2:
        match_data = filtered_df.groupby('ismatch').size().reset_index(name='counts')
        match_data['status'] = match_data['ismatch'].map({True: 'Matches', False: 'Non-Matches'})

        fig_pie = px.pie(match_data, values='counts', names='status', title='Match vs Non-Match Percentage')
        st.plotly_chart(fig_pie)

## Prompts with most number of re-runs
if not df_tasks.empty and not df_executions.empty and not df_stepruns.empty:
    st.header("Which Prompts Have the Highest Re-Runs?")

    executions_with_prompts = df_executions.merge(df_tasks[['req_id', 'prompt', 'level']], on='req_id')

    re_run_counts_by_execution = df_stepruns.groupby('execution_id').size().reset_index(name='re_run_count')

    executions_with_re_runs = executions_with_prompts.merge(re_run_counts_by_execution, on='execution_id')

    max_re_runs_per_prompt = (executions_with_re_runs.groupby('prompt')
                              .apply(lambda x: x.nlargest(1, 're_run_count'))
                              .reset_index(drop=True))

    top_re_runs_prompts = max_re_runs_per_prompt.sort_values(by='re_run_count', ascending=False).head(5)

    top_re_runs_prompts['file_present'] = top_re_runs_prompts['prompt'].apply(
        lambda x: 'Yes' if not df[df['question'] == x]['file_name'].isnull().all() else 'No'
    )

    st.table(top_re_runs_prompts[['prompt', 'level', 're_run_count', 'file_present']])
else:
    st.error("No data available to display charts.")