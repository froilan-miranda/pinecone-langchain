import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = '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'

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

davinci = OpenAI(model_name='text-davinci-003')

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(question))

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
llm_chain.generate(qs)

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))
