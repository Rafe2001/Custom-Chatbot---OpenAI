import streamlit as st 
import json
import os
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader,ServiceContext
from langchain import OpenAI
import openai



def construct_index(directory_path):
     max_input_size=4096
     num_outputs = 2000
     max_chunk_overlap = 20
     chunk_size_limit = 600

     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit = chunk_size_limit)

     llm_predictor = LLMPredictor(llm=OpenAI(temperature = 0.5, model_name = "text-curie-001",max_tokens=num_outputs))

     documents = SimpleDirectoryReader(directory_path).load_data()

     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
     index = GPTSimpleVectorIndex.from_documents(documents, service_context = service_context)

     index.save_to_disk('index2.json')
     return index

def main():
    st.title("Tag8 Chatbot")
    st.header("This is Tag8's chatbot answering your Queries")
    os.environ['OPENAI_API_KEY'] = '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'

    if st.button("Construct Index"):
        with st.spinner("Constructing Index..."):
            construct_index('Data')
            st.success("Index constructed successfully!")

    query = st.text_input("Ask a question:")
    if st.button("Ask AI"):
        index = GPTSimpleVectorIndex.load_from_disk('index2.json')
        response = index.query(query)
        st.markdown(f"Response: {response.response}")

if __name__ == "__main__":
    main()
    
    
    
    
