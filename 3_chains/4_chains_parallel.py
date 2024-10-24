import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu eres un experto revisor de producto."),
        ("human", "Lista las principales características del producto {producto_nombre}."),
    ]
)

def analyze_pros(caracteristicas):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Tu eres un experto revisor de producto."),
            (
                "human",
                "Dadas estas características: {caracteristicas}, lista los pro de estas características.",
            ),
        ]
    )
    return pros_template.format_prompt(caracteristicas=caracteristicas)

def analyze_cons(caracteristicas):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Tu eres un experto revisor de producto."),
            (
                "human",
                "Dadas estas características: {caracteristicas}, lista los cons de estas características.",
            ),
        ]
    )
    return cons_template.format_prompt(caracteristicas=caracteristicas)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"producto_nombre": "Ford Maverick"})

print(result)