import os
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

# first initialize the large language model
llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003',
    openai_api_key='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'

)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)
print(conversation.prompt.template)
print('\n')

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
print(conversation_buf("Good morning AI!"))


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

print(
    count_tokens(
        conversation_buf, 
        "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
    )
)
print(
    count_tokens(
        conversation_buf,
        "I just want to analyze the different possibilities. What can you think of?"
    )
)

print(
    count_tokens(
        conversation_buf, 
        "Which data source types could be used to give context to the model?"
    )
)

print(
    count_tokens(
        conversation_buf, 
        "What is my aim again?"
    )
)

print(conversation_buf.memory.buffer)
