import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /audio/speech

## **LiteLLM Python SDK Usage**
### Quick Start 

```python
from pathlib import Path
from litellm import speech
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
        model="openai/tts-1",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
    )
response.stream_to_file(speech_file_path)
```

### Async Usage 

```python
from litellm import aspeech
from pathlib import Path
import os, asyncio

os.environ["OPENAI_API_KEY"] = "sk-.."

async def test_async_speech(): 
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await litellm.aspeech(
            model="openai/tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        )
    response.stream_to_file(speech_file_path)

asyncio.run(test_async_speech())
```

## **LiteLLM Proxy Usage**

LiteLLM provides an openai-compatible `/audio/speech` endpoint for Text-to-speech calls.

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

**Setup**

```bash
- model_name: tts
  litellm_params:
    model: openai/tts-1
    api_key: os.environ/OPENAI_API_KEY
```

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```
## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| OpenAI      |   [Usage](#quick-start)                 |
| Azure OpenAI|   [Usage](../docs/providers/azure#azure-text-to-speech-tts)                 |
| Azure Speech Services |   [Usage](#azure-speech-services-text-to-speech)                 |
| Vertex AI   |   [Usage](../docs/providers/vertex#text-to-speech-apis)                 |
| Gemini      |   [Usage](#gemini-text-to-speech)                 |

## `/audio/speech` to `/chat/completions` Bridge

LiteLLM allows you to use `/chat/completions` models to generate speech through the `/audio/speech` endpoint. This is useful for models like Gemini's TTS-enabled models that are only accessible via `/chat/completions`.

## Azure Speech Services Text-to-Speech

LiteLLM supports Azure Speech Services (Cognitive Services) for direct text-to-speech conversion. This is different from Azure OpenAI TTS and provides more voice options and advanced SSML features.

### Python SDK Usage

```python showLineNumbers title="Azure Speech Services SDK Usage"
import litellm
import os
from pathlib import Path

# Set your Azure Speech Services credentials
os.environ["AZURE_SPEECH_KEY"] = "your-speech-services-key"
os.environ["AZURE_SPEECH_ENDPOINT"] = "https://eastus.tts.speech.microsoft.com"

def test_azure_speech_services():
    result = litellm.speech(
        model="azure_speech/eastus",  # Region as model
        input="Hello world, this is Azure Speech Services!",
        voice="en-US-AriaNeural",  # Azure neural voice
        api_key=os.getenv("AZURE_SPEECH_KEY"),
        api_base=os.getenv("AZURE_SPEECH_ENDPOINT"),
    )
    
    # Save to file
    speech_file_path = Path(__file__).parent / "azure_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_azure_speech_services()
```

### Advanced SSML Usage

```python showLineNumbers title="Azure Speech Services with SSML"
import litellm
import os

# SSML input for advanced speech control
ssml_input = '''
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="en-US-AriaNeural">
        <prosody rate="+20%" pitch="+10%">
            Hello! This is a demonstration of Azure Speech Services
        </prosody>
        <break time="1s"/>
        <prosody rate="-10%">
            with advanced SSML features.
        </prosody>
    </voice>
</speak>
'''

result = litellm.speech(
    model="azure_speech/eastus",
    input=ssml_input,
    voice="en-US-AriaNeural",  # Will be overridden by SSML voice
    api_key=os.getenv("AZURE_SPEECH_KEY"),
    optional_params={
        "response_format": "audio-24khz-48kbitrate-mono-mp3"
    }
)
```

### Speed Control

```python showLineNumbers title="Azure Speech Services with Speed Control"
import litellm

# Speed control (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
result = litellm.speech(
    model="azure_speech/eastus",
    input="This speech will be faster than normal",
    voice="en-US-JennyNeural",
    api_key=os.getenv("AZURE_SPEECH_KEY"),
    optional_params={
        "speed": 1.5  # 50% faster
    }
)
```

### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Azure Speech Services Proxy Configuration"
model_list:
- model_name: azure-speech-tts
  litellm_params:
    model: azure_speech/eastus
    api_key: os.environ/AZURE_SPEECH_KEY
    api_base: https://eastus.tts.speech.microsoft.com
```

**Make Request:**

```bash showLineNumbers title="Azure Speech Services TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azure-speech-tts",
    "input": "Hello from Azure Speech Services!",
    "voice": "en-US-AriaNeural"
  }' \
  --output azure_speech.mp3
```

### Available Voices

Azure Speech Services supports hundreds of neural voices across many languages. Popular English voices include:

- `en-US-AriaNeural` - Female, conversational
- `en-US-JennyNeural` - Female, customer service
- `en-US-GuyNeural` - Male, conversational  
- `en-US-DavisNeural` - Male, professional
- `en-GB-SoniaNeural` - Female, British
- `en-AU-NatashaNeural` - Female, Australian

For a complete list, see the [Azure Speech Services documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#neural-voices).

### Response Formats

Supported audio formats include:

- `audio-16khz-128kbitrate-mono-mp3` (default)
- `audio-24khz-48kbitrate-mono-mp3`
- `audio-48khz-96kbitrate-mono-mp3`
- `riff-24khz-16bit-mono-pcm`
- `audio-24khz-160kbitrate-mono-mp3`

### Gemini Text-to-Speech

LiteLLM supports Gemini TTS models through two approaches:

1. **Speech-to-completion bridge** (default) - Uses `/chat/completions` with audio modalities
2. **Direct TTS API** (new) - Direct audio generation endpoint

#### Bridge Approach (Default)

```python showLineNumbers title="Gemini Text-to-Speech SDK Usage"
import litellm
import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

def test_audio_speech_gemini():
    result = litellm.speech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Save to file
    from pathlib import Path
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_gemini()
```

#### Direct TTS Approach (New)

```python showLineNumbers title="Gemini Direct TTS Usage"
import litellm
import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

def test_gemini_direct_tts():
    result = litellm.speech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="This uses the direct TTS endpoint for faster audio generation",
        voice="alloy",  # Voice style
        api_key=os.getenv("GEMINI_API_KEY"),
        optional_params={
            "use_direct_tts": True,  # Enable direct TTS mode
            "response_format": "mp3"
        }
    )
    
    # Save to file  
    from pathlib import Path
    speech_file_path = Path(__file__).parent / "gemini_direct_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_gemini_direct_tts()
```

#### Voice Configuration

```python showLineNumbers title="Gemini TTS with Advanced Voice Config"
import litellm

# Advanced voice configuration for direct TTS
voice_config = {
    "voice": "nova",
    "speed": 1.2,
    "language": "en-US",
    "style": "conversational"
}

result = litellm.speech(
    model="gemini/gemini-2.5-flash-preview-tts",
    input="Advanced voice configuration example",
    voice=voice_config,
    optional_params={"use_direct_tts": True}
)
```

#### Async Usage

```python showLineNumbers title="Gemini Text-to-Speech Async Usage"
import litellm
import asyncio
import os
from pathlib import Path

os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

async def test_async_gemini_speech():
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    response = await litellm.aspeech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    response.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

asyncio.run(test_async_gemini_speech())
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Gemini Proxy Configuration"
model_list:
- model_name: gemini-tts
  litellm_params:
    model: gemini/gemini-2.5-flash-preview-tts
    api_key: os.environ/GEMINI_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="Gemini TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output gemini_speech.mp3
```

### Vertex AI Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="Vertex AI Text-to-Speech SDK Usage"
import litellm
import os
from pathlib import Path

# Set your Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account.json"

def test_audio_speech_vertex():
    result = litellm.speech(
        model="vertex_ai/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
    )
    
    # Save to file
    speech_file_path = Path(__file__).parent / "vertex_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_vertex()
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Vertex AI Proxy Configuration"
model_list:
- model_name: vertex-tts
  litellm_params:
    model: vertex_ai/gemini-2.5-flash-preview-tts
    vertex_project: your-project-id
    vertex_location: us-central1
```

**Make Request:**

```bash showLineNumbers title="Vertex AI TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "en-US-Wavenet-D"
  }' \
  --output vertex_speech.mp3
```

## ✨ Enterprise LiteLLM Proxy - Set Max Request File Size 

Use this when you want to limit the file size for requests sent to `audio/transcriptions`

```yaml
- model_name: whisper
  litellm_params:
    model: whisper-1
    api_key: sk-*******
    max_file_size_mb: 0.00001 # 👈 max file size in MB  (Set this intentionally very small for testing)
  model_info:
    mode: audio_transcription
```

Make a test Request with a valid file
```shell
curl --location 'http://localhost:4000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--form 'file=@"/Users/ishaanjaffer/Github/litellm/tests/gettysburg.wav"' \
--form 'model="whisper"'
```


Expect to see the follow response 

```shell
{"error":{"message":"File size is too large. Please check your file size. Passed file size: 0.7392807006835938 MB. Max file size: 0.0001 MB","type":"bad_request","param":"file","code":500}}%  
```