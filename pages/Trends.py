import streamlit as st

title = '<p style="font-family: Sans Serif; color:white; text-align: center; font-size: 45px;">Confusion Matrix</p>'
st.markdown(title, unsafe_allow_html=True)

st.image("confmatrix.png")
