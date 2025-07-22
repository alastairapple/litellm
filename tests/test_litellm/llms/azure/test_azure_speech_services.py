"""
Tests for Azure Speech Services TTS provider.
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import litellm
from litellm.llms.azure.azure_speech_services import AzureSpeechServices


class TestAzureSpeechServices:
    """Test the Azure Speech Services TTS implementation."""

    def test_build_ssml_simple_text(self):
        """Test SSML building with simple text."""
        azure_speech = AzureSpeechServices()
        
        result = azure_speech._build_ssml(
            input="Hello world",
            voice="en-US-AriaNeural",
            optional_params={},
        )
        
        expected = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            '<voice name="en-US-AriaNeural">'
            'Hello world'
            '</voice>'
            '</speak>'
        )
        assert result == expected

    def test_build_ssml_with_speed(self):
        """Test SSML building with speed parameter."""
        azure_speech = AzureSpeechServices()
        
        result = azure_speech._build_ssml(
            input="Hello world",
            voice="en-US-AriaNeural",
            optional_params={"speed": 1.5},
        )
        
        expected = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            '<voice name="en-US-AriaNeural">'
            '<prosody rate="+50%">'
            'Hello world'
            '</prosody>'
            '</voice>'
            '</speak>'
        )
        assert result == expected

    def test_build_ssml_existing_ssml(self):
        """Test SSML building when input is already SSML."""
        azure_speech = AzureSpeechServices()
        
        existing_ssml = '<speak><voice name="test">Hello</voice></speak>'
        result = azure_speech._build_ssml(
            input=existing_ssml,
            voice="en-US-AriaNeural",
            optional_params={},
        )
        
        # Should return the existing SSML unchanged
        assert result == existing_ssml

    def test_build_speech_request(self):
        """Test building the Azure Speech Services request."""
        azure_speech = AzureSpeechServices()
        
        headers, ssml_content, url = azure_speech._build_speech_request(
            input="Hello world",
            voice="en-US-AriaNeural",
            optional_params={},
            api_key="test-key",
            api_base=None,
            model="eastus",
        )
        
        assert url == "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
        assert headers["Ocp-Apim-Subscription-Key"] == "test-key"
        assert headers["Content-Type"] == "application/ssml+xml"
        assert "Hello world" in ssml_content

    def test_build_speech_request_with_custom_base(self):
        """Test building request with custom API base."""
        azure_speech = AzureSpeechServices()
        
        headers, ssml_content, url = azure_speech._build_speech_request(
            input="Hello world",
            voice="en-US-AriaNeural",
            optional_params={},
            api_key="test-key",
            api_base="https://custom.speech.service.com/",
            model="custom",
        )
        
        assert url == "https://custom.speech.service.com/cognitiveservices/v1"

    @pytest.mark.asyncio
    async def test_audio_speech_mocked_success(self):
        """Test successful audio speech generation with mocked response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio data"
        
        with patch('litellm.llms.custom_httpx.http_handler._get_httpx_client') as mock_client:
            mock_client.return_value.post.return_value = mock_response
            
            azure_speech = AzureSpeechServices()
            result = azure_speech.audio_speech(
                model="eastus",
                input="Hello world",
                voice="en-US-AriaNeural",
                optional_params={},
                api_key="test-key",
                api_base=None,
                max_retries=1,
                timeout=30,
            )
            
            assert result is not None
            # Verify the request was made
            mock_client.return_value.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_audio_speech_mocked_success(self):
        """Test successful async audio speech generation with mocked response."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio data"
        
        with patch('litellm.llms.custom_httpx.http_handler.get_async_httpx_client') as mock_client:
            mock_client.return_value.post.return_value = mock_response
            
            azure_speech = AzureSpeechServices()
            result = await azure_speech.async_audio_speech(
                model="eastus",
                input="Hello world",
                voice="en-US-AriaNeural",
                optional_params={},
                api_key="test-key",
                api_base=None,
                max_retries=1,
                timeout=30,
            )
            
            assert result is not None
            # Verify the async request was made
            mock_client.return_value.post.assert_called_once()


@pytest.mark.asyncio
async def test_litellm_azure_speech_services():
    """Test Azure Speech Services through litellm.speech()."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake audio data"
    
    with patch('litellm.llms.custom_httpx.http_handler._get_httpx_client') as mock_client:
        mock_client.return_value.post.return_value = mock_response
        
        result = litellm.speech(
            model="azure_speech/eastus",
            input="Hello world",
            voice="en-US-AriaNeural",
            api_key="test-key",
        )
        
        assert result is not None
        # Verify the request was made
        mock_client.return_value.post.assert_called_once()
        
        # Check that the correct URL was used
        call_args = mock_client.return_value.post.call_args
        assert "eastus.tts.speech.microsoft.com" in call_args.kwargs["url"]


@pytest.mark.asyncio 
async def test_litellm_azure_speech_services_async():
    """Test Azure Speech Services through litellm.aspeech()."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"fake audio data"
    
    with patch('litellm.llms.custom_httpx.http_handler.get_async_httpx_client') as mock_client:
        mock_client.return_value.post.return_value = mock_response
        
        result = await litellm.aspeech(
            model="azure_speech/eastus",
            input="Hello world",
            voice="en-US-AriaNeural",
            api_key="test-key",
        )
        
        assert result is not None
        # Verify the async request was made
        mock_client.return_value.post.assert_called_once()