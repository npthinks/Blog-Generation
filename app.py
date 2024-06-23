import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getLlamaresponse(input_text, words_count, blog_niche):
    ## Llama 2 model
    llm = CTransformers(model = "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type = "llama",
                        config = {"max_new_tokens" : 256,
                                  "temperature" : 0.01})
    ## Prompt template
    template="""
        Write a blog for {blog_niche} job profile for a topic {input_text}
        within {words_count} words.
            """

    prompt = PromptTemplate(input_variables= ["blog_niche", "input_text", "words_count"],
                            template= template)
    
    ## Response
    response = llm(prompt.format(blog_niche = blog_niche, input_text = input_text, words_count = words_count))

    print(response)
    return response

## Page configuration

st.set_page_config(page_title= "Blog Generation",
                  page_icon= "#np",
                  layout= "centered",
                  initial_sidebar_state= "collapsed")

# Page heading
st.header("Generate Blogs")

#Page columns
input_text = st.text_input("Blog Topic")

column_1, column_2 = st.columns([5,5])

with column_1:
    words_count = st.text_input("Number of Words")

with column_2:
    blog_niche = st.selectbox("Blog Niche",
                              ("Data Science", 
                              "Artificial Intelligence",
                              "General Topic"), index=0)
    
submit = st.button("Generate Blog")

# Blog generation
if submit:
    st.write(getLlamaresponse(input_text, words_count, blog_niche))
