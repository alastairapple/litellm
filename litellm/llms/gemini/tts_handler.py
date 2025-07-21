"""
Direct Gemini Text-to-Speech implementation.

This provides a direct audio generation endpoint for Gemini models,
as an alternative to the speech-to-completion bridge.
"""

import base64
import json
from typing import Any, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    _get_httpx_client
)
from litellm.types.llms.openai import HttpxBinaryResponseContent

from .google_genai.vertex_and_google_ai_studio_gemini import VertexLLM


class GeminiTTSHandler(VertexLLM):
    """
    Direct Gemini Text-to-Speech handler.
    
    This provides a direct audio generation endpoint for Gemini models
    that support TTS, without using the speech-to-completion bridge.
    """

    def __init__(self):
        super().__init__()

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, dict]] = None,
        optional_params: Optional[dict] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 300,
        logging_obj: Optional[Any] = None,
        _is_async: bool = False,
        **kwargs,
    ) -> HttpxBinaryResponseContent:
        """
        Generate speech directly from Gemini TTS models.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash-preview-tts")
            input: Text to convert to speech
            voice: Voice configuration (dict or string)
            optional_params: Additional parameters
            api_key: Gemini API key
            api_base: API base URL
            timeout: Request timeout
            logging_obj: Logging object
            _is_async: Whether this is an async request

        Returns:
            HttpxBinaryResponseContent: Audio data response
        """
        if _is_async:
            return self.async_audio_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                logging_obj=logging_obj,
                **kwargs,
            )  # type: ignore

        # Build the request
        headers, request_data, url = self._build_audio_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params or {},
            api_key=api_key,
            api_base=api_base,
        )

        # Log the request
        if logging_obj:
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": request_data,
                    "api_base": url,
                    "headers": headers,
                },
            )

        # Make the request
        sync_handler = _get_httpx_client()
        response = sync_handler.post(
            url=url,
            headers=headers,
            json=request_data,
            timeout=timeout,
        )

        if response.status_code != 200:
            error_msg = f"Gemini TTS request failed with status {response.status_code}: {response.text}"
            raise Exception(error_msg)

        # Process the response
        return self._process_audio_response(response, model)

    async def async_audio_speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, dict]] = None,
        optional_params: Optional[dict] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 300,
        logging_obj: Optional[Any] = None,
        **kwargs,
    ) -> HttpxBinaryResponseContent:
        """
        Generate speech asynchronously from Gemini TTS models.
        """
        # Build the request
        headers, request_data, url = self._build_audio_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=optional_params or {},
            api_key=api_key,
            api_base=api_base,
        )

        # Log the request
        if logging_obj:
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": request_data,
                    "api_base": url,
                    "headers": headers,
                },
            )

        # Make the async request
        async_handler = get_async_httpx_client(llm_provider="gemini")
        response = await async_handler.post(
            url=url,
            headers=headers,
            json=request_data,
            timeout=timeout,
        )

        if response.status_code != 200:
            error_msg = f"Gemini TTS request failed with status {response.status_code}: {response.text}"
            raise Exception(error_msg)

        # Process the response
        return self._process_audio_response(response, model)

    def _build_audio_request(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, dict]],
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
    ) -> tuple[dict, dict, str]:
        """
        Build the Gemini audio generation request.

        Returns:
            tuple: (headers, request_data, url)
        """
        # Clean model name
        clean_model = model.replace("gemini/", "")

        # Build URL
        if api_base:
            base_url = api_base.rstrip("/")
        else:
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            
        url = f"{base_url}/models/{clean_model}:generateContent"

        # Build headers
        headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            url += f"?key={api_key}"

        # Build request data for audio generation
        request_data = {
            "contents": [
                {
                    "role": "user", 
                    "parts": [
                        {"text": input}
                    ]
                }
            ],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": optional_params.get("temperature", 0.0),
            }
        }

        # Add voice configuration if provided
        if voice:
            if isinstance(voice, str):
                # Simple string voice - use as style parameter
                request_data["generationConfig"]["audioStyle"] = voice
            elif isinstance(voice, dict):
                # Dict voice - merge into generation config
                request_data["generationConfig"].update(voice)

        # Add response modalities for audio
        request_data["generationConfig"]["responseModalities"] = ["AUDIO"]

        # Add audio format configuration
        audio_format = optional_params.get("response_format", "mp3")
        if audio_format == "mp3":
            request_data["generationConfig"]["responseAudioFormat"] = "MP3"
        elif audio_format == "wav":
            request_data["generationConfig"]["responseAudioFormat"] = "WAV"
        else:
            request_data["generationConfig"]["responseAudioFormat"] = "MP3"  # Default

        return headers, request_data, url

    def _process_audio_response(
        self, 
        response: httpx.Response, 
        model: str
    ) -> HttpxBinaryResponseContent:
        """
        Process the Gemini audio response.

        Args:
            response: HTTP response from Gemini
            model: Model name for format detection

        Returns:
            HttpxBinaryResponseContent: Processed audio response
        """
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            raise Exception(f"Failed to parse JSON response: {response.text}")

        # Extract audio data from response
        candidates = response_data.get("candidates", [])
        if not candidates:
            raise Exception("No candidates found in Gemini TTS response")

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Find audio part
        audio_part = None
        for part in parts:
            if "inlineData" in part and part["inlineData"].get("mimeType", "").startswith("audio/"):
                audio_part = part
                break

        if not audio_part:
            raise Exception("No audio data found in Gemini TTS response")

        # Extract base64 audio data
        audio_data_b64 = audio_part["inlineData"]["data"]
        mime_type = audio_part["inlineData"]["mimeType"]

        # Decode base64 to binary
        binary_data = base64.b64decode(audio_data_b64)

        # Create response with appropriate headers
        response_headers = {
            "Content-Type": mime_type,
            "Content-Length": str(len(binary_data)),
        }

        # Create an httpx.Response object with the audio data
        audio_response = httpx.Response(
            status_code=200,
            content=binary_data,
            headers=response_headers,
        )

        return HttpxBinaryResponseContent(response=audio_response)