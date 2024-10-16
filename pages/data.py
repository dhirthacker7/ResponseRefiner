import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from validator import df


if not df.empty:
     st.header("Database Results")
     st.dataframe(df)