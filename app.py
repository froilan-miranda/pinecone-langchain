import os
from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about NFL 2010
print(llm_chain.run(question))

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
res = llm_chain.generate(qs)
res

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=flan_t5
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))
