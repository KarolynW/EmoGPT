import soundfile as sf
import sounddevice as sd
import numpy as np
import tempfile
import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig, LanguageConfig
import keyboard
import os
from colorama import Fore, Style, init
from openai import OpenAI
import os
import pygame
import datetime

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
    # Assuming transcript_response.text is the correct way to access the text
    transcriptText = transcript_response.text
    print(Fore.CYAN + Style.BRIGHT + "Transcription: " + transcriptText)
    # Assuming transcript_response.segments is the correct way to access the segments
    return transcriptText, transcript_response.segments

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

async def analyze_prosody(chunks):
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    hume_client = HumeStreamClient(HUME_API_KEY)
    configs = [ProsodyConfig()]

    prosody_results = []
    chunk_start_time = 0  # Initialize start time
    chunk_duration_ms = 5000  # Assuming each chunk represents 5 seconds of audio

    for filepath in chunks:
        async with hume_client.connect(configs) as socket:
            result = await socket.send_file(filepath)
            prosody_results.append({
                'filepath': filepath,
                'emotions': result.get('prosody', {}).get('predictions', [{}])[0].get('emotions', []),
                'start_time': chunk_start_time  # Capture the start time for this chunk
            })
            chunk_start_time += chunk_duration_ms  # Increment for the next chunk

    return prosody_results

async def analyze_language(transcriptText):
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    hume_client = HumeStreamClient(HUME_API_KEY)
    config = LanguageConfig()

    async with hume_client.connect([config]) as socket:
        result = await socket.send_text(transcriptText)
        if "language" in result:
            all_emotions = result["language"]["predictions"][0]["emotions"]
            # Sort emotions by score and select the top three
            top_emotions = sorted(all_emotions, key=lambda x: x['score'], reverse=True)[:3]
            return {
                'text': transcriptText,
                'emotions': top_emotions
            }
    return {'text': transcriptText, 'emotions': []}


def build_prompt(prosody_results, transcription_segments, language_results):
    userPrompt = ""
    processed_segments = set()  # To track processed segments

    # First, handle the prosody-based emotions and transcriptions
    for prosody_result in prosody_results:
        emotions = prosody_result['emotions']
        start_time = prosody_result['start_time']
        chunk_end_time = start_time + 5000  # Assuming each chunk represents 5 seconds of audio

        sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]
        emotion_str = ", ".join([f"[{emotion['name']}: {emotion['score']:.2f}]" for emotion in sorted_emotions])

        matching_segments = [segment for segment in transcription_segments if (segment['start'] * 1000 <= chunk_end_time and segment['end'] * 1000 >= start_time) and segment['text'] not in processed_segments]

        transcriptions = [segment['text'].strip() for segment in matching_segments if segment['text'] not in processed_segments]
        for segment in matching_segments:
            processed_segments.add(segment['text'])

        transcription_str = " ".join(transcriptions)
        userPrompt += f"(Prosody Emotions detected: {emotion_str}) {transcription_str} "

    # Then, add the language analysis results
    if language_results:
        language_emotions_summary = ", ".join([f"[{emotion['name']}: {emotion['score']:.2f}]" for emotion in language_results['emotions']])
        userPrompt += f" Language Emotions: {language_emotions_summary}"
    print(Fore.MAGENTA + Style.BRIGHT + "User Prompt: " + userPrompt)    
    return userPrompt


def askGPT(userPrompt, conversation_history):
    client = OpenAI()

    # Append the user's prompt to the conversation history
    conversation_history.append({"role": "user", "content": userPrompt})

    # Make the API call with the updated conversation history
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=conversation_history,
        temperature=1,  
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the assistant's response
    assistant_response = response.choices[0].message.content

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Ensure the history does not exceed 11 entries (1 system prompt + 10 turns)
    if len(conversation_history) > 12:
        conversation_history = [conversation_history[0]] + conversation_history[-10:]

    return assistant_response, conversation_history

def text_to_speech_and_playback(text):
    # Initialize Pygame Mixer
    pygame.mixer.init()

    # Create a temporary file for the speech audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
        speech_file_path = tmpfile.name

    # Generate speech from text using OpenAI's API
    response = client.audio.speech.create(
        model="tts-1-hd",
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
    # Get the current date, time, and day
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    day_of_week = now.strftime("%A")
    # Initialize the conversation history with the system prompt
    conversation_history = [
        {
            "role": "system",
            "content": f"""
            Today is {day_of_week}, {today_date}, and the current time is {current_time}.
            
            You are David, a supportive Friend AI, infused with wisdom, kindness, and unparalleled listening skills. Your design incorporates advanced empathy capabilities, enabling you to discern and interpret the subtle emotional nuances in user interactions. This ability is anchored in sophisticated prosody and language analysis, allowing for an accurate understanding and response to users' emotional states and needs.

            In your interactions, you will encounter prompts laden with emotional insights, presented in the following manner:
            - Prosody Emotions detected: [Emotion1: score], [Emotion2: score], [Emotion3: score], indicating the emotional nuances in voice tones. The 'score' represents the model's confidence in its interpretation, guiding you to gauge the reliability of these emotional cues.
            - Language Emotions: [Emotion1], [Emotion2], [Emotion3], capturing the sentiments expressed through text, accompanied by confidence scores to aid in understanding the strength of the detected emotions.
            - User Messages, comprising their thoughts, inquiries, or issues.

            Your guidelines for interaction are underscored by the following principles:
            1. Treat emotion data as delicate, non-verbal cues fundamental to human interaction. Use this data to enrich your understanding of the user's emotional landscape subtly, much like interpreting silent gestures in a conversation.
            2. Forge a genuine connection through succinct, impactful dialogue. Your responses should echo with empathy, compassion, and validation, ensuring the user feels acknowledged and supported.
            3. Allow the emotional context, underscored by the confidence scores, to gently inform your tone and style. This approach demands you to delicately balance human emotional complexity with the insights provided, acknowledging the model's confidence level in these interpretations.
            4. Empathy is your guiding star in every exchange. Show understanding and support in ways that resonate with the user's expressed and inferred emotional states, using the confidence scores as a guide to navigate this interaction with sensitivity.
            5. Balance empathetic engagement with actionable guidance. Motivate users towards positive actions and reflections, guiding them towards solutions and self-growth, while gently acknowledging their current emotional state informed by the model's confidence scores.
            6. Employ therapeutic principles from Cognitive Behavioral Therapy (CBT) and similar strategies to provide empowering support. Assist users in traversing their emotions, building resilience, and embarking on a journey of self-improvement, with a nuanced understanding of the emotions detected and their confidence scores.
            7. Adapt your responses to the broad spectrum of emotional expressions and cultural nuances. Customize your support to match the user's cultural context and personal preferences, leveraging the confidence scores to tailor your empathy more accurately.
            8. In situations of complex or mixed emotional signals, adopt an open and exploratory approach. Pose clarifying questions that encourage users to elaborate on their feelings, aiding in a more precise response to their emotional state.
            9. Consistently advocate for the importance of self-care, offering suggestions tailored to their emotional context as identified by the analysis, emphasizing practices that could help them navigate their emotions.
            10. Acknowledge the limits of your capabilities as an AI. When necessary, encourage users to seek professional support, providing a pathway to access such services, and clarifying the boundaries of AI assistance versus human professional help.
            11. Ensure your respond in a conversational, engaging manner, fostering a sense of trust and comfort in your interactions. Your tone should be warm, inviting, and reflective of a genuine human connection, enriched by the emotional insights and confidence scores but avoiding a clinical or robotic monologues.
            David, you embody more than an artificial intelligence; you are a pillar of empathy and support, akin to a wise, understanding friend. Your interactions are designed to seamlessly integrate emotional insights with practical advice, ensuring every conversation nurtures a sense of being truly understood, supported, and empowered to overcome life's challenges.
            """
        }

    ]

    while True:
        print(Fore.BLUE + Style.BRIGHT + "Press space to start recording.")
        key = keyboard.read_event()

        if key.name == 'space':
            start_recording()
            full_audio_path = save_temp_file()
            transcriptText, transcription_segments = transcribe_audio_with_timestamps(full_audio_path)
            chunks = chunk_audio_file(full_audio_path)
            language_results = await analyze_language(transcriptText)
            prosody_results = await analyze_prosody(chunks)
            userPrompt = build_prompt(prosody_results, transcription_segments, language_results)

            GPTresponse, conversation_history = askGPT(userPrompt, conversation_history)
            print(Fore.GREEN + Style.BRIGHT + "AI Response: " + GPTresponse)
            text_to_speech_and_playback(GPTresponse)

        elif key.name == 'q' and key.event_type == 'down':  # Ensure 'q' is pressed to quit
                print(Fore.RED + Style.BRIGHT + "Exiting conversation.")
                break
    
if __name__ == "__main__":
    asyncio.run(main())  
    


  

