from unittest.mock import MagicMock, patch

from psysafe.utils.llm_parsing import call_llm, get_client, get_llm_response


@patch("psysafe.utils.llm_parsing.ai.Client")
def test_get_client(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    client = get_client({"openai": {"api_key": "test"}})
    mock_client_class.assert_called_once()
    mock_client.configure.assert_called_once_with({"openai": {"api_key": "test"}})
    assert client == mock_client


@patch("psysafe.utils.llm_parsing.get_client")
def test_call_llm(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = call_llm("openai:gpt-4o", [{"role": "user", "content": "hello"}])

    mock_client.chat.completions.create.assert_called_once_with(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    assert response == mock_response


@patch("psysafe.utils.llm_parsing.call_llm")
def test_get_llm_response(mock_call_llm):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mocked text"))]
    mock_call_llm.return_value = mock_response

    result = get_llm_response("openai:gpt-4o", "hello", system_prompt="Sys")

    mock_call_llm.assert_called_once_with(
        model="openai:gpt-4o",
        messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "hello"}],
        temperature=0.7,
        client=None,
    )
    assert result == "Mocked text"
