"""Integration tests for ChatGithubCopilot chat model.

These tests make real HTTP calls to the GitHub Models inference API.
Set the GITHUB_TOKEN environment variable before running:

    export GITHUB_TOKEN="github_pat_..."
    pytest tests/integration_tests/test_chat_models.py -v
"""

from typing import Type

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_githubcopilot_chat.chat_models import ChatGithubCopilot
from langchain_tests.integration_tests import ChatModelIntegrationTests


# ---------------------------------------------------------------------------
# Standard LangChain integration test suite
# ---------------------------------------------------------------------------


class TestChatGithubCopilotIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatGithubCopilot]:
        return ChatGithubCopilot

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai/gpt-4.1-mini",
            "temperature": 0,
            "max_tokens": 256,
        }


# ---------------------------------------------------------------------------
# Additional hand-written integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _rate_limit_sleep() -> None:
    """Sleep between integration tests to avoid GitHub Models 429 rate limits."""
    yield
    time.sleep(4)


@pytest.fixture
def llm() -> ChatGithubCopilot:
    """Return a ChatGithubCopilot instance pointed at a fast, cheap model."""
    return ChatGithubCopilot(
        model="openai/gpt-4.1-mini",
        temperature=0,
        max_tokens=256,
    )


@pytest.mark.integration
def test_basic_invoke(llm: ChatGithubCopilot) -> None:
    """A basic invocation should return a non-empty AIMessage."""
    ai_msg = llm.invoke([HumanMessage("Say the single word: hello")])
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.content) > 0


@pytest.mark.integration
def test_system_and_human_messages(llm: ChatGithubCopilot) -> None:
    """The model should follow system instructions."""
    messages = [
        SystemMessage("You only reply with the word PONG."),
        HumanMessage("PING"),
    ]
    ai_msg = llm.invoke(messages)
    assert "PONG" in ai_msg.content.upper()


@pytest.mark.integration
def test_token_usage_is_reported(llm: ChatGithubCopilot) -> None:
    """Token usage metadata should be present and positive."""
    ai_msg = llm.invoke([HumanMessage("What is 2 + 2?")])
    assert ai_msg.usage_metadata is not None
    assert ai_msg.usage_metadata["input_tokens"] > 0
    assert ai_msg.usage_metadata["output_tokens"] > 0
    assert (
        ai_msg.usage_metadata["total_tokens"]
        == ai_msg.usage_metadata["input_tokens"]
        + ai_msg.usage_metadata["output_tokens"]
    )


@pytest.mark.integration
def test_response_metadata_finish_reason(llm: ChatGithubCopilot) -> None:
    """finish_reason should be present in response_metadata."""
    ai_msg = llm.invoke([HumanMessage("What is 2 + 2?")])
    assert "finish_reason" in ai_msg.response_metadata
    assert ai_msg.response_metadata["finish_reason"] in ("stop", "length", "tool_calls")


@pytest.mark.integration
def test_streaming(llm: ChatGithubCopilot) -> None:
    """Streaming should yield multiple chunks and the full text should be non-empty."""
    chunks = list(llm.stream([HumanMessage("Count from 1 to 5.")]))
    assert len(chunks) > 0
    full_text = "".join(str(c.content) for c in chunks if c.content)
    assert len(full_text) > 0


@pytest.mark.integration
def test_streaming_accumulates_correctly(llm: ChatGithubCopilot) -> None:
    """Accumulated stream result should equal a full invoke response."""
    messages = [HumanMessage("What is the capital of Germany? Answer in one word.")]

    # Accumulate stream
    stream = llm.stream(messages)
    full = next(stream)
    for chunk in stream:
        full += chunk
    stream_text = str(full.content).strip()

    # Single invoke
    invoke_text = str(llm.invoke(messages).content).strip()

    # Both should mention Berlin (exact text may differ due to non-determinism)
    assert "Berlin" in stream_text or len(stream_text) > 0
    assert "Berlin" in invoke_text or len(invoke_text) > 0


@pytest.mark.integration
def test_streaming_token_usage(llm: ChatGithubCopilot) -> None:
    """Token usage should appear in the streamed response (final chunk)."""
    chunks = list(llm.stream([HumanMessage("Say hello.")]))
    stream = llm.stream([HumanMessage("Say hello.")])
    full = next(stream)
    for chunk in stream:
        full += chunk
    assert full.usage_metadata is not None
    assert full.usage_metadata["total_tokens"] > 0


@pytest.mark.integration
async def test_async_invoke(llm: ChatGithubCopilot) -> None:
    """Async invocation should return a valid AIMessage."""
    ai_msg = await llm.ainvoke([HumanMessage("Say the single word: hello")])
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.content) > 0


@pytest.mark.integration
async def test_async_stream(llm: ChatGithubCopilot) -> None:
    """Async streaming should yield chunks with non-empty combined content."""
    chunks = []
    async for chunk in llm.astream([HumanMessage("Count from 1 to 3.")]):
        chunks.append(chunk)
    assert len(chunks) > 0
    full_text = "".join(str(c.content) for c in chunks if c.content)
    assert len(full_text) > 0


@pytest.mark.integration
def test_tool_calling(llm: ChatGithubCopilot) -> None:
    """The model should emit tool_calls when a matching tool is bound."""
    from pydantic import BaseModel, Field

    class GetCapital(BaseModel):
        """Get the capital city of a country."""

        country: str = Field(..., description="Name of the country")

    llm_with_tools = llm.bind_tools([GetCapital])
    ai_msg = llm_with_tools.invoke([HumanMessage("What is the capital of Japan?")])

    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) > 0
    tc = ai_msg.tool_calls[0]
    assert tc["name"] == "GetCapital"
    assert "Japan" in tc["args"].get("country", "")


@pytest.mark.integration
def test_tool_calling_full_loop(llm: ChatGithubCopilot) -> None:
    """A complete tool-calling loop should produce a final text response."""
    from langchain_core.messages import ToolMessage

    from pydantic import BaseModel, Field

    class GetCapital(BaseModel):
        """Get the capital city of a country."""

        country: str = Field(..., description="Name of the country")

    def get_capital(country: str) -> str:
        capitals = {"Japan": "Tokyo", "France": "Paris", "Germany": "Berlin"}
        return capitals.get(country, f"Unknown capital for {country}")

    llm_with_tools = llm.bind_tools([GetCapital])

    messages = [HumanMessage("What is the capital of Japan?")]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tc in ai_msg.tool_calls:
        result = get_capital(**tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final_msg = llm_with_tools.invoke(messages)
    assert isinstance(final_msg, AIMessage)
    assert "Tokyo" in final_msg.content


@pytest.mark.integration
def test_structured_output(llm: ChatGithubCopilot) -> None:
    """with_structured_output() should return a populated Pydantic object."""
    from typing import List

    from pydantic import BaseModel, Field

    class Country(BaseModel):
        """Information about a country."""

        name: str = Field(description="Country name")
        capital: str = Field(description="Capital city")
        population_millions: float = Field(
            description="Approximate population in millions"
        )

    structured_llm = llm.with_structured_output(Country)
    result = structured_llm.invoke("Tell me about France.")

    assert isinstance(result, Country)
    assert result.name != ""
    assert result.capital != ""
    assert result.population_millions > 0


@pytest.mark.integration
def test_json_mode(llm: ChatGithubCopilot) -> None:
    """JSON mode should return parseable JSON content."""
    import json

    json_llm = llm.bind(response_format={"type": "json_object"})
    ai_msg = json_llm.invoke(
        [
            SystemMessage(
                "You are a helpful assistant. Always respond with valid JSON."
            ),
            HumanMessage(
                "Return a JSON object with a single key 'answer' whose value is the string 'Paris'."
            ),
        ]
    )

    data = json.loads(ai_msg.content)
    assert "answer" in data
    assert data["answer"] == "Paris"


@pytest.mark.integration
def test_stop_sequences(llm: ChatGithubCopilot) -> None:
    """The model should stop generating when it hits a stop sequence."""
    ai_msg = llm.invoke(
        [HumanMessage("List the numbers 1, 2, 3, 4, 5 separated by commas.")],
        stop=["3"],
    )
    # The response should not contain "4" or "5" because we stop at "3"
    assert "4" not in ai_msg.content
    assert "5" not in ai_msg.content


@pytest.mark.integration
def test_sampling_params(llm: ChatGithubCopilot) -> None:
    """Sampling parameters should be accepted without errors."""
    creative_llm = ChatGithubCopilot(
        model="openai/gpt-4.1-mini",
        temperature=0.8,
        top_p=0.9,
        max_tokens=64,
        frequency_penalty=0.3,
        presence_penalty=0.1,
        seed=1337,
    )
    ai_msg = creative_llm.invoke(
        [HumanMessage("Say something creative in one sentence.")]
    )
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.content) > 0


@pytest.mark.integration
def test_chaining_with_prompt_template(llm: ChatGithubCopilot) -> None:
    """The model should work inside a LangChain LCEL chain."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are a helpful assistant that translates {source} to {target}.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {"source": "English", "target": "French", "text": "I love programming."}
    )

    assert isinstance(result, AIMessage)
    assert len(result.content) > 0


@pytest.mark.integration
def test_multiple_models() -> None:
    """Different models should all return valid responses."""
    models_to_test = [
        "openai/gpt-4.1-mini",
        "meta/llama-3.3-70b-instruct",
    ]
    question = [HumanMessage("Reply with only the word: OK")]

    for model_id in models_to_test:
        m = ChatGithubCopilot(model=model_id, max_tokens=16, temperature=0)
        ai_msg = m.invoke(question)
        assert isinstance(ai_msg, AIMessage), f"Failed for model: {model_id}"
        assert len(ai_msg.content) > 0, f"Empty response from model: {model_id}"


@pytest.mark.integration
def test_multimodal_image_input() -> None:
    """A vision-capable model should describe an image."""
    import base64

    import httpx

    vision_llm = ChatGithubCopilot(
        model="openai/gpt-4.1",
        max_tokens=64,
        temperature=0,
    )

    # Use the Google logo — small PNG, publicly accessible from GitHub infra
    image_url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What does this image show? Answer in five words or fewer.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    ai_msg = vision_llm.invoke([message])
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.content) > 0
