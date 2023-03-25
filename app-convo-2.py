from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.callbacks import get_openai_callback

llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003',
    openai_api_key='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'

)

conversation_sum = ConversationChain(
	llm=llm,
	memory=ConversationSummaryMemory(llm=llm)
)

print(conversation_sum.memory.prompt.template)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

print(
    count_tokens(
        conversation_sum, 
        "Good morning AI!"
    )
)

print(
    count_tokens(
        conversation_sum, 
        "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
    )
)

print(
    count_tokens(
        conversation_sum, 
        "I just want to analyze the different possibilities. What can you think of?"
    )
)

print(
    count_tokens(
        conversation_sum, 
        "Which data source types could be used to give context to the model?"
    )
)

print(
    count_tokens(
        conversation_sum, 
        "What is my aim again?"
    )
)

print(conversation_sum.memory.buffer)
