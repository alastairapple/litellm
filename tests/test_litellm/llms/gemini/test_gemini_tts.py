"""
Test Gemini TTS (Text-to-Speech) functionality
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.utils import get_supported_openai_params


class TestGeminiTTSTransformation:
    """Test Gemini TTS transformation functionality"""

    def test_gemini_tts_model_detection(self):
        """Test that TTS models are correctly identified"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test TTS models
        assert config.is_model_gemini_audio_model("gemini-2.5-flash-preview-tts") == True
        assert config.is_model_gemini_audio_model("gemini-2.5-pro-preview-tts") == True
        
        # Test non-TTS models
        assert config.is_model_gemini_audio_model("gemini-2.5-flash") == False
        assert config.is_model_gemini_audio_model("gemini-2.5-pro") == False
        assert config.is_model_gemini_audio_model("gpt-4o-audio-preview") == False

    def test_gemini_tts_supported_params(self):
        """Test that audio parameter is included for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test TTS model
        params = config.get_supported_openai_params("gemini-2.5-flash-preview-tts")
        assert "audio" in params
        
        # Test that other standard params are still included
        assert "temperature" in params
        assert "max_tokens" in params
        assert "modalities" in params
        
        # Test non-TTS model
        params_non_tts = config.get_supported_openai_params("gemini-2.5-flash")
        assert "audio" not in params_non_tts

    def test_gemini_tts_audio_parameter_mapping(self):
        """Test audio parameter mapping for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {
                "voice": "Kore",
                "format": "pcm16"
            }
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Check speech config is created
        assert "speechConfig" in result
        assert "voiceConfig" in result["speechConfig"]
        assert "prebuiltVoiceConfig" in result["speechConfig"]["voiceConfig"]
        assert result["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
        
        # Check response modalities
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_audio_parameter_with_existing_modalities(self):
        """Test audio parameter mapping when modalities already exist"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {
                "voice": "Puck",
                "format": "pcm16"
            }
        }
        optional_params = {
            "responseModalities": ["TEXT"]
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Check that AUDIO is added to existing modalities
        assert "responseModalities" in result
        assert "TEXT" in result["responseModalities"]
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_no_audio_parameter(self):
        """Test that non-audio parameters are handled normally"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should not have speech config
        assert "speechConfig" not in result
        # Should not automatically add audio modalities
        assert "responseModalities" not in result

    def test_gemini_tts_invalid_audio_parameter(self):
        """Test handling of invalid audio parameter"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": "invalid_string"  # Should be dict
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should not create speech config for invalid audio param
        assert "speechConfig" not in result

    def test_gemini_tts_empty_audio_parameter(self):
        """Test handling of empty audio parameter"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {}
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should still set response modalities even with empty audio config
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_audio_format_validation(self):
        """Test audio format validation for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test invalid format
        non_default_params = {
            "audio": {
                "voice": "Kore",
                "format": "wav"  # Invalid format
            }
        }
        optional_params = {}
        
        with pytest.raises(ValueError, match="Unsupported audio format for Gemini TTS models"):
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="gemini-2.5-flash-preview-tts",
                drop_params=False
            )

    def test_gemini_tts_utils_integration(self):
        """Test integration with LiteLLM utils functions"""
        # Test that get_supported_openai_params works with TTS models
        params = get_supported_openai_params("gemini-2.5-flash-preview-tts", "gemini")
        assert "audio" in params
        
        # Test non-TTS model
        params_non_tts = get_supported_openai_params("gemini-2.5-flash", "gemini")
        assert "audio" not in params_non_tts


def test_gemini_tts_completion_mock():
    """Test Gemini TTS completion with mocked response"""
    with patch('litellm.completion') as mock_completion:
        # Mock a successful TTS response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated audio response"
        mock_completion.return_value = mock_response
        
        # Test completion call with audio parameter
        response = litellm.completion(
            model="gemini-2.5-flash-preview-tts",
            messages=[{"role": "user", "content": "Say hello"}],
            audio={"voice": "Kore", "format": "pcm16"}
        )
        
        assert response is not None
        assert response.choices[0].message.content is not None


if __name__ == "__main__":
    pytest.main([__file__])


# Additional tests for direct Gemini TTS implementation

from litellm.llms.gemini.tts_handler import GeminiTTSHandler


class TestGeminiTTSDirectHandler:
    """Test the direct Gemini TTS implementation."""

    def test_build_audio_request(self):
        """Test building the Gemini audio request."""
        gemini_tts = GeminiTTSHandler()
        
        headers, request_data, url = gemini_tts._build_audio_request(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice="alloy",
            optional_params={},
            api_key="test-key",
            api_base=None,
        )
        
        assert "generativelanguage.googleapis.com" in url
        assert "key=test-key" in url
        assert headers["Content-Type"] == "application/json"
        assert request_data["contents"][0]["parts"][0]["text"] == "Hello world"
        assert request_data["generationConfig"]["audioStyle"] == "alloy"
        assert "AUDIO" in request_data["generationConfig"]["responseModalities"]

    def test_build_audio_request_with_voice_dict(self):
        """Test building request with voice as dictionary."""
        gemini_tts = GeminiTTSHandler()
        
        voice_config = {
            "voice": "alloy",
            "speed": 1.2,
            "language": "en-US"
        }
        
        headers, request_data, url = gemini_tts._build_audio_request(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice=voice_config,
            optional_params={},
            api_key="test-key",
            api_base=None,
        )
        
        # Voice dict should be merged into generation config
        assert request_data["generationConfig"]["voice"] == "alloy"
        assert request_data["generationConfig"]["speed"] == 1.2
        assert request_data["generationConfig"]["language"] == "en-US"

    def test_build_audio_request_with_response_format(self):
        """Test building request with custom response format."""
        gemini_tts = GeminiTTSHandler()
        
        headers, request_data, url = gemini_tts._build_audio_request(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice="alloy",
            optional_params={"response_format": "wav"},
            api_key="test-key",
            api_base=None,
        )
        
        assert request_data["generationConfig"]["responseAudioFormat"] == "WAV"

    def test_process_audio_response(self):
        """Test processing a successful Gemini audio response."""
        gemini_tts = GeminiTTSHandler()
        
        # Mock response data
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/mp3",
                                    "data": "ZmFrZSBhdWRpbyBkYXRh"  # base64 for "fake audio data"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        
        result = gemini_tts._process_audio_response(mock_response, "gemini-2.5-flash-preview-tts")
        
        assert result is not None
        # The response should contain the decoded audio data

    @pytest.mark.asyncio
    async def test_audio_speech_mocked_success(self):
        """Test successful audio speech generation with mocked response."""
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/mp3",
                                    "data": "ZmFrZSBhdWRpbyBkYXRh"  # base64 for "fake audio data"
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        
        with patch('litellm.llms.custom_httpx.http_handler._get_httpx_client') as mock_client:
            mock_client.return_value.post.return_value = mock_response
            
            gemini_tts = GeminiTTSHandler()
            result = gemini_tts.audio_speech(
                model="gemini-2.5-flash-preview-tts",
                input="Hello world",
                voice="alloy",
                api_key="test-key",
            )
            
            assert result is not None
            # Verify the request was made
            mock_client.return_value.post.assert_called_once()


@pytest.mark.asyncio
async def test_litellm_gemini_direct_tts():
    """Test direct Gemini TTS through litellm.speech()."""
    response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/mp3",
                                "data": "ZmFrZSBhdWRpbyBkYXRh"  # base64 for "fake audio data"
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = response_data
    
    with patch('litellm.llms.custom_httpx.http_handler._get_httpx_client') as mock_client:
        mock_client.return_value.post.return_value = mock_response
        
        result = litellm.speech(
            model="gemini/gemini-2.5-flash-preview-tts",
            input="Hello world",
            voice="alloy",
            api_key="test-key",
            optional_params={"use_direct_tts": True},
        )
        
        assert result is not None
        # Verify the request was made
        mock_client.return_value.post.assert_called_once()
        
        # Check that the correct URL was used
        call_args = mock_client.return_value.post.call_args
        assert "generativelanguage.googleapis.com" in call_args.kwargs["url"]
