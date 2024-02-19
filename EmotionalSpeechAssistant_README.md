
# Emotional Speech Assistant

This project is an innovative application that leverages advanced AI technologies to analyze emotions in spoken language, generate empathetic responses, and convert these responses into speech. It integrates various OpenAI models and other technologies to create a seamless interaction experience.

## Features

- **Voice Recording**: Capture user's voice input through a microphone.
- **Emotion Analysis**: Utilize the Hume AI to analyze the emotional content of the spoken word.
- **AI-Generated Responses**: Leverage OpenAI's GPT models to generate responses based on analyzed emotions.
- **Text-to-Speech**: Convert the AI's text responses into audible speech.
- **Temporary File Handling**: Manage temporary audio files for a clean, efficient application run.

## Dependencies

- OpenAI
- Hume AI
- Pygame for audio playback
- SoundFile and SoundDevice for audio recording and file handling
- Asyncio for asynchronous programming
- Colorama for colored terminal output

## Installation

Ensure you have Python 3.7+ installed. Clone the repository, and install the required packages:

```bash
pip install openai hume-sdk pygame soundfile sounddevice colorama
```

## Usage

Run the application from the terminal:

```bash
python main.py
```

Follow the on-screen instructions to start recording. The application will guide you through recording, emotion analysis, AI response generation, and playback.

## Configuration

Make sure to set your OpenAI API key and Hume API key in your environment variables:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
export HUME_API_KEY='your_hume_api_key_here'
```

## Contributing

Contributions are welcome! Feel free to open a pull request or an issue if you have suggestions or encounter any problems.

## License

Distributed under the MIT License. See `LICENSE` for more information.
