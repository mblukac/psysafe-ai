from psysafe.drivers.anthropic import AnthropicChatDriver


def test_anthropic_driver_metadata_never_exposes_credentials() -> None:
    driver = AnthropicChatDriver(
        model="test-claude",
        api_key="sk-ant-secret",
        base_url="https://private.example",
    )

    assert driver.get_metadata() == {
        "driver_type": "anthropic",
        "model_name": "test-claude",
        "supports_streaming": True,
    }
