import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Load and preprocess the music dataset"""
    df = pd.read_csv("data/muse_v3.csv")
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']
    df = df[['name','emotional','pleasant','link','artist']]
    df = df.sort_values(by=["emotional", "pleasant"])
    
    # Split by emotion categories
    df_sad = df[:18000]
    df_fear = df[18000:36000]
    df_angry = df[36000:54000]
    df_neutral = df[54000:72000]
    df_happy = df[72000:]
    
    return {
        'sad': df_sad,
        'fear': df_fear,
        'angry': df_angry,
        'neutral': df_neutral,
        'happy': df_happy
    }