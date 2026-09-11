
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="ministral-8b-latest", mistral_api_key= os.getenv("MISTRAL_API_KEY"), temperature=0.3)


def split_transcript(transcript: str)-> list:
    splitter= RecursiveCharacterTextSplitter(
        chunk_size= 3000,
        chunk_overlap= 200
    )
    
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a video transcript concisely. "
            "Focus on the main ideas, facts, arguments, and important details."
        ),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            summary = map_chain.invoke({"text": chunk})
            chunk_summaries.append(summary)

        except Exception as e:
            print(f"⚠️ Failed to summarize chunk {i + 1}: {e}")

    # If Mistral failed completely
    if not chunk_summaries:
        return (
            "Summary generation is temporarily unavailable because "
            "the AI API rate limit was reached."
        )

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert video summarizer. "
            "Combine these partial summaries into one concise, "
            "professional summary using bullet points."
        ),
        ("human", "{text}"),
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    try:
        return combined_chain.invoke({"text": combined})

    except Exception as e:
        print(f"⚠️ Final summary generation failed: {e}")

        # Return the partial summaries instead of crashing
        return combined

def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the transcript, generate a short professional title "
            "of maximum 8 words. Return only the title."
        ),
        ("human", "{text}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    try:
        return title_chain.invoke({
            "text": transcript[:2000]
        })

    except Exception as e:
        print(f"⚠️ Title generation failed: {e}")
        return "AI Economy and Technology Discussion"