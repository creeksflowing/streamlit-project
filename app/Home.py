import streamlit as st
from utils import head

"""
st.set_page_config(
    page_title=[' Wave(r) - Impulse Response Module'],
    page_icon='assets/acoustics.png'
)
"""

head()


if st.button('Impulse Response'):
    result = add(1, 2)
    st.write('result: %s' % result)
else:
    st.write('Goodbye')
