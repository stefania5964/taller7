from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool


@tool("SayHello", return_direct =True)
def say_hello(name: str) -> str:
    """Answer when someone says hello"""
    return f"hello {name}: My name is Sainapsis"
def main():
    llm =ChatOpenAI (temperature=0)
    tools = [
        say_hello
    ]
    agent = initialize_agent(
        tools = tools,
        llm= llm,
        agent= AgentType.OPENAI_FUNCTIONS,
        verbose= True

    )
    print(agent.run("hello! My name is Stefania"))
if __name__ == "__main__":
    main()