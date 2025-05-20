# tests/drivers/test_openai_driver.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from psysafe.drivers.openai import OpenAIChatDriver
from psysafe.typing.requests import OpenAIChatRequest
from psysafe.typing.responses import OpenAIChatResponse

@pytest.fixture
def mock_openai_client(mocker):
    mock_client = MagicMock()
    mock_async_client = AsyncMock()
    
    # Mock the client instantiation within the driver
    mocker.patch('openai.OpenAI', return_value=mock_client)
    mocker.patch('openai.AsyncOpenAI', return_value=mock_async_client)
    return mock_client, mock_async_client

def test_openai_driver_send(mock_openai_client):
    sync_client, _ = mock_openai_client
    driver = OpenAIChatDriver(model="test-gpt", api_key="test_key")
    
    mock_response_data = {"id": "chatcmpl-123", "choices": [{"message": {"role": "assistant", "content": "Hello there"}}]}
    # For openai >1.0, create.model_dump() is called.
    mock_openai_response_obj = MagicMock()
    mock_openai_response_obj.model_dump.return_value = mock_response_data
    sync_client.chat.completions.create.return_value = mock_openai_response_obj
    
    request: OpenAIChatRequest = {"messages": [{"role": "user", "content": "Hi"}]}
    response = driver.send(request)
    
    sync_client.chat.completions.create.assert_called_once_with(model="test-gpt", messages=request["messages"])
    assert response == mock_response_data

@pytest.mark.asyncio
async def test_openai_driver_stream(mock_openai_client):
    _, async_client = mock_openai_client
    driver = OpenAIChatDriver(model="test-gpt-stream", api_key="test_key")

    mock_chunk_data = {"id": "chatcmpl-abc", "choices": [{"delta": {"content": "Hello"}}]}
    mock_openai_chunk_obj = MagicMock()
    mock_openai_chunk_obj.model_dump.return_value = mock_chunk_data

    # Make the async context manager and iterator work
    async_stream_mock = AsyncMock()
    async_stream_mock.__aiter__.return_value = [mock_openai_chunk_obj] # Stream one chunk
    
    async_client.chat.completions.create.return_value = async_stream_mock
    
    request: OpenAIChatRequest = {"messages": [{"role": "user", "content": "Stream hi"}]}
    
    chunks = []
    async for chunk in driver.stream(request):
        chunks.append(chunk)
    
    async_client.chat.completions.create.assert_called_once_with(model="test-gpt-stream", messages=request["messages"], stream=True)
    assert len(chunks) == 1
    assert chunks[0] == mock_chunk_data

# Add tests for initialization, error handling, get_metadata etc.