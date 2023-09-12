#Bring in deps
import os
from apikey import apikey

import wikipedia
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜️🔗 Youtube GPT Creator')
prompt = st.text_input('Enter a prompt for the AI to complete:')

# Prompt Templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a Youtube video title about {topic}'
    )

discription_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a Youtube video discription on: {title}, while leveraging the following research: {wikipedia_research}'
    )

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
discription_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


#llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
discription_chain = LLMChain(llm=llm, prompt=discription_template, verbose=True, output_key='discription', memory=discription_memory)

wiki = WikipediaAPIWrapper()

#Show output to the screen if there is a prompt
if prompt:
    #Run the title chain
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    #Run the discription chain
    discription = discription_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(discription) 
    
    with st.expander('title History'):
        st.info(title_memory.buffer)

    with st.expander('discription History'):
        st.info(discription_memory.buffer)

    with st.expander('wikipedia research'):
        st.info(wiki_research)