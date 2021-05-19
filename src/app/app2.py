import streamlit as st # type:ignore
def app():
    st.title('Prédictions et résultats des differents models utlisés')
    st.write('Welcome to app2')
    method = st.sidebar.radio('modeles', ['Régression logistique', 'XGBoost', 'Random Forest', 'SVM', 'Reseau de neurones'])
    
    
    if method == 'Régression logistique' : 
        st.write('valeur de l\'accuracy')
        st.image('/home/apprenant/simplon_project/Brief-Emotion-Analysis-Text/images/logistic_regression/step1_data1_results.png')
    elif method == 'XGBoost' :
        st.write('XGBoost')
    elif method == 'Random Forest' :
        st.write('Random Forest')
    elif method == 'SVM' :
        st.write('SVM')
    else : 
        st.write('Reseau de neurones')