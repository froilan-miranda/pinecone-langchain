from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback

llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003',
    openai_api_key=os.environ.get('OPENAI_TOKEN')
)

conversation_bufw = ConversationChain(
	llm=llm,
	memory=ConversationBufferWindowMemory(k=1)
)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

print(
    count_tokens(
        conversation_bufw, 
        "Good morning AI!"
    )
)

print(
    count_tokens(
        conversation_bufw, 
        "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
    )
)

print(
    count_tokens(
        conversation_bufw, 
        "I just want to analyze the different possibilities. What can you think of?"
    )
)

print(
    count_tokens(
        conversation_bufw, 
        "Which data source types could be used to give context to the model?"
    )
)

print(
    count_tokens(
        conversation_bufw, 
        "What is my aim again?"
    )
)

bufw_history = conversation_bufw.memory.load_memory_variables(
    inputs=[]
)['history']

print(bufw_history)
