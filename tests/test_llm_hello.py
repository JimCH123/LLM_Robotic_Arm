"""Roadmap 0.4: check that the OpenAI API key works by asking the model to say hello."""

import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # read OPENAI_API_KEY and OPENAI_MODEL from the .env file


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set in .env")
def test_llm_says_hello():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    reply = llm.invoke("Reply with exactly one word: hello")
    print("Model reply:", reply.text)
    assert "hello" in reply.text.lower()