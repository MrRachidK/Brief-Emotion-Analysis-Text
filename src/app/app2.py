import streamlit as st

def app():
    st.title('Prédictions et résultats des differents models utlisés')
    st.write('Welcome to app2')
    method = st.sidebar.radio('modeles', ['Régression logistique', 'XGBoost', 'Random Forest', 'SVM', 'Reseau de neurones'])
    
    
    if method == 'Régression logistique' : 
        st.write('Régression logistique')
    elif method == 'XGBoost' :
        st.write('XGBoost')
    elif method == 'Random Forest' :
        st.write('Random Forest')
    elif method == 'SVM' :
        st.write('SVM')
    else : 
        st.write('Reseau de neurones')