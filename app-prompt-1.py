import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
os.environ['OPENAI_API_KEY'] = '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'

# initialize the models
openai = OpenAI(
    model_name="text-davinci-003",
    # openai_api_key="YOUR_API_KEY"
)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

print(
    openai(
        prompt_template.format(
            query="Which libraries and model providers offer LLMs?"
        )
    )
)
