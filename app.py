## imports
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import gradio as gr
import os

HF_TOKEN = os.environ.get("HF_TOKEN", None)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. 
    Original question: {question}""",
)
PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">

   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Powered by MistralxNomic</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask your PDF anything...</p>
</div>
"""
css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

## Function to process pdf related question
def process(query='What is this about?',hist=None,local_path=None):
    if local_path:
        loader = UnstructuredPDFLoader(file_path=local_path)
        data = loader.load()
    else:
        print("Upload a PDF file")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag"
    )
    local_model = "mistral"
    llm = ChatOllama(model=local_model)
    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
    )
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    reply = chain.invoke(query)
    print(reply)
    return reply


# questionBox = gr.Textbox(label="Enter your question:", placeholder="Type something here...")
title = '<img src="https://github.com/EternalBlissard/AskPdfAnything/blob/main/src/AskPdfAnything_transparent.png" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> '
description = "An LLM based model to make question answering with your pdfs a bit easy"
article = "Created by [Eternal Bliassard](https://github.com/EternalBlissard)."
chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')
# demo = gr.Interface(fn=predictionMaker, 
#                 additional_inputs=[questionBox,gr.File()], 
#                 examples=exampleList, 
#                 title=title,
#                 description=description,
#                 article=article)
# with gr,Blocks(fill)
with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(title)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    gr.ChatInterface(
        fn=process,
        chatbot=chatbot,
        # fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.File(),
            # gr.Slider(minimum=0,
            #         maximum=1, 
            #         step=0.1,
            #         value=0.95, 
            #         label="Temperature", 
            #         render=False),
            # gr.Slider(minimum=128, 
            #         maximum=4096,
            #         step=1,
            #         value=512, 
            #         label="Max new tokens", 
            #         render=False ),
            ],
        examples=[
            ['What is this about?'],
            ['Is this a novel method?'],
            ['Is there anything wrong with the approach?'],
            ['Who is the antagonist?'],
            ['What is the moral?']
            ],
        cache_examples=False,
                    )
    
    gr.Markdown(description)
    gr.Markdown(article)
    # gr.Markdown(LICENSE)
    
# if __name__ == "__main__":
#     demo.launch()
# Launch the demo!
demo.launch(share=True) 


