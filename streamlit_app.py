import streamlit as st
from langchain_helper import get_agent
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Amorphous solid dispersions analyst 💊")
st.markdown("""
### Example questions
- Ask me: *how many APIs are in the dataset*  
- Ask me: *Is carbamazepine stable at 40°C and 75% RH?*  
- Ask me: *to make plots*  
""")
##--Use Streamlit resource caching to prevent reloading gemini every page refresh
@st.cache_resource
def load_everything():
    return get_agent()

agent, df = load_everything()

question = st.text_input("Question: ")

if question:
    with st.spinner("Analysing dataset..."):
        try:
            response = agent.run(question, additional_args={
                'df': df,
                'plt':plt,
                'sns':sns,
            }
                                )
        
            st.write(f"You asked: {question}")
            
            if plt.get_fignums():
                st.write("### 📊 Generated Visualization")
                st.pyplot(plt.gcf())
                plt.clf()

            st.success(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")