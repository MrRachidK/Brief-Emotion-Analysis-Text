import streamlit as st # type: ignore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    #--------------------------------------------#
    #            Import data                     #
    #--------------------------------------------#
    ef_brut = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/01_raw/Emotion_final.csv')
    ef_clean = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_emotion_final.csv')
    te_brut = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/01_raw/text_emotion.csv')
    te_clean = pd.read_csv('/home/apprenant/Documents/Brief-Emotion-Analysis-Text/data/02_cleaned/cleaned_text_emotion.csv')

    #--------------------------------------------#
    #                Header                      #
    #--------------------------------------------#
    st.title('Dataframes et les graphiques')
    #--------------------------------------------#
    #                Sidebar                     #
    #--------------------------------------------#
    st.sidebar.title('show dataframes')
    method = st.sidebar.radio('dataframe', ['ef_brut', 'ef_clean', 'te_brut', 'te_clean'])
    user_value = st.sidebar.text_input('recherche')
    st.sidebar.slider('donnez le nombre de valeurs lié à la recherche', min_value = 1 , max_value = 10)

    st.markdown('les resultats s\'affichent ici')
    #--------------------------------------------#
    #                  user_value                #
    #--------------------------------------------#
    if method == 'ef_brut' : 
        st.dataframe(ef_brut)
        
    elif method == 'ef_clean': 
        st.dataframe(ef_clean)
        fig2 = plt.figure()
        ax1 = sns.countplot(x = "label", data = ef_clean)
        plt.xticks(rotation = 45)
        st.pyplot(fig2)

    elif method == 'te_brut' : 
        st.dataframe(te_brut)

    elif method == 'te_clean': 
        st.dataframe(te_clean)
        fig1 = plt.figure()
        ax = sns.countplot(x = "label", data = te_clean)
        plt.xticks(rotation = 45)
        st.pyplot(fig1)

