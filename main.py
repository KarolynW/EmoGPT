import soundfile as sf
import sounddevice as sd
import numpy as np
import tempfile
import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig
import keyboard
import os
from colorama import Fore, Style, init
from openai import OpenAI
import os
import pygame

client=OpenAI()

# Initialize Colorama
init(autoreset=True)

SAMPLE_RATE = 44100
recording = []

def callback(indata, frames, time, status):
    recording.extend(indata.copy())

def start_recording():
    global recording
    recording = []
    print(Fore.BLUE + Style.BRIGHT + "Recording... Press space again to stop.")
    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=2, dtype='float32'):
        keyboard.wait('space')
    print(Fore.GREEN + Style.BRIGHT + "Recording stopped.")

def save_temp_file():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='temp_audio_', mode='w+b')
    sf.write(temp_file.name, np.array(recording), SAMPLE_RATE)
    print(Fore.YELLOW + Style.BRIGHT + f"Audio saved to {temp_file.name}")
    return temp_file.name

def transcribe_audio_with_timestamps(filepath):
    with open(filepath, "rb") as audio_file:
        transcript_response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    transcriptText = transcript_response.text
    print(Fore.CYAN + Style.BRIGHT + "Transcription: " + transcriptText)
    
    # Directly return the segments attribute of the transcription response
    return transcript_response.segments
  
def chunk_audio_file(filepath, chunk_duration_ms=5000):
    with sf.SoundFile(filepath) as sound_file:
        frames = int(SAMPLE_RATE * (chunk_duration_ms / 1000))
        total_frames = sound_file.frames
        chunks = []
        for start in range(0, total_frames, frames):
            chunk_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='chunk_', mode='w+b')
            chunk_frames = min(frames, total_frames - start)
            sound_file.seek(start)
            chunk_data = sound_file.read(chunk_frames)
            sf.write(chunk_temp_file.name, chunk_data, SAMPLE_RATE)
            chunks.append(chunk_temp_file.name)
        print(Fore.CYAN + Style.BRIGHT + "Audio chunked for analysis.")
        return chunks

async def analyze_emotion_and_build_prompt(chunks, transcription_segments):
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    hume_client = HumeStreamClient(HUME_API_KEY)
    configs = [ProsodyConfig()]

    userPrompt = ""  # Initialize the userPrompt string
    chunk_duration_ms = 5000
    chunk_start_time = 0

    for index, filepath in enumerate(chunks):
        async with hume_client.connect(configs) as socket:
            result = await socket.send_file(filepath)
            chunk_end_time = chunk_start_time + chunk_duration_ms

            if index == len(chunks) - 1:
                with sf.SoundFile(filepath) as sound_file:
                    chunk_end_time = chunk_start_time + int((sound_file.frames / SAMPLE_RATE) * 1000)

            # Find segments that overlap with this chunk
            matching_segments = [segment for segment in transcription_segments if (segment['start'] * 1000 <= chunk_end_time and segment['end'] * 1000 >= chunk_start_time)]

            # Aggregate transcriptions for this chunk
            transcriptions = " ".join([segment['text'].strip() for segment in matching_segments])

            if 'predictions' in result.get('prosody', {}):
                emotions = result['prosody']['predictions'][0]['emotions']
                sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]  # Top 3 emotions
                emotion_str = ", ".join([f"[{emotion['name']}: {emotion['score']:.2f}]" for emotion in sorted_emotions])
                userPrompt += f"(Emotions detected: {emotion_str}) {transcriptions} "

            chunk_start_time += chunk_duration_ms

    print(Fore.CYAN + Style.BRIGHT + "Emotion analysis complete. User prompt built.")
    print(Fore.YELLOW + Style.BRIGHT + "User prompt: " + userPrompt)
    return userPrompt


def askGPT(userPrompt):

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
            "role": "system",
            "content": """You are a supportive Friend AI. You are wise, kind and always ready to listen. 
                You have enhanced empathy skills - you will be supplied with prompt from the user in this format:

                (Emotions detected: [Emotion1: importance], [Emotion2: importance], [Emotion3: importance]) USER MESSAGE 1 (Emotions detected: [Emotion1: importance], [Emotion2: importance], [Emotion3: importance]) USER MESSAGE 2 etc.

                You must:

                1. The emotion information is for you and you only as additional context to your conversation.
                2. You should user the user emotion indications to inform the tone and style of your responses. Remember that the emotion analysis may not always be accurate so be subtle.
                3. You should be empathetic to the participant - you should always aim to improve the users wellbeing subtly
                """
            },
            {
            "role": "user",
            "content": userPrompt
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    answer = response.choices[0].message.content
    return answer

def text_to_speech_and_playback(text):
    # Initialize Pygame Mixer
    pygame.mixer.init()

    # Create a temporary file for the speech audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
        speech_file_path = tmpfile.name

    # Generate speech from text using OpenAI's API
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )

    # Stream the response to the temporary file
    response.stream_to_file(speech_file_path)

    # Load the audio file and play it
    pygame.mixer.music.load(speech_file_path)
    pygame.mixer.music.play()

    # Wait for the playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Cleanup: No need to manually delete the file as NamedTemporaryFile with delete=True will handle it  

async def main():
    print(Fore.BLUE + Style.BRIGHT + "Press space to start recording.")
    keyboard.wait('space')
    start_recording()
    full_audio_path = save_temp_file()

    transcription_segments = transcribe_audio_with_timestamps(full_audio_path)
    chunks = chunk_audio_file(full_audio_path)
    userPrompt = await analyze_emotion_and_build_prompt(chunks, transcription_segments)

    GPTresponse = askGPT(userPrompt)
    print(Fore.GREEN + Style.BRIGHT + "AI Response: " + GPTresponse)  

    # Convert AI response to speech and play it back
    text_to_speech_and_playback(GPTresponse)


if __name__ == "__main__":
    asyncio.run(main())

  

