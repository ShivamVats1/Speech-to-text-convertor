import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Text:", text)
        return text
    except sr.UnknownValueError:
        print("Speech not understood")
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")

def transcribe_audio(audio_file):
    # Load pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    # Load audio
    speech, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values

    # Perform inference
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the prediction
    transcription = processor.decode(predicted_ids[0])
    print("Transcription:", transcription)
    return transcription

def save_text_to_file(text, filename="output.txt"):
    with open(filename, "w") as f:
        f.write(text)
    print(f"Transcription saved to {filename}")

# Example usage
audio_file_path = r"C:\Users\Admin\Desktop\SPEECHTOTEXT\audio_files\test.wav"  # Use double backslashes or raw string
text = transcribe_audio(audio_file_path)
if text:
    save_text_to_file(text)
