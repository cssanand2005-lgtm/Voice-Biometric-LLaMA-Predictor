import os
import numpy as np
import gradio as gr
import torch
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cdist
import requests

# 1. Initialize the SpeechBrain Speaker Verification Model
print("Loading stable speaker verification model...")
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain_spkrec")
    )
except Exception as e:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# In-memory speaker database
voice_database = {}

def get_audio_embedding(audio_path):
    # 🚨 COMPLETELY BYPASS TORCHAUDIO FILE LOADING TO AVOID TORCHCODEC ERRORS
    # Read raw audio array data directly from disk using the pure python soundfile library
    data, sample_rate = sf.read(audio_path)
    
    # Convert multi-channel audio arrays to a single mono array matrix
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Convert the raw data into a basic float32 PyTorch tensor manually
    waveform = torch.FloatTensor(data).unsqueeze(0) # Shape: [1, time_steps]
    
    # Manually handle resampling if the browser recorded at a frequency other than 16kHz
    if sample_rate != 16000:
        # A simple mathematical linear interpolation resample that doesn't need torchaudio
        old_indices = np.arange(len(data))
        new_indices = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sample_rate))
        resampled_data = np.interp(new_indices, old_indices, data)
        waveform = torch.FloatTensor(resampled_data).unsqueeze(0)
        
    # Extract the voice print features cleanly purely using basic matrix tensors
    with torch.no_grad():
        embeddings = classifier.encode_batch(waveform)
        embedding_np = embeddings.squeeze().cpu().numpy()
        
    return embedding_np

# 2. Function to enroll a user's voice
def enroll_voice(name, audio_path):
    if not name or not audio_path:
        return "Please provide both a name and an audio recording."
    try:
        embedding = get_audio_embedding(audio_path)
        voice_database[name] = embedding
        return f"Successfully enrolled {name}'s voice print!"
    except Exception as e:
        return f"Error during enrollment: {str(e)}"

# 3. Function to query local LLaMA via Ollama API
def ask_llama(speaker_name, confidence):
    url = "http://localhost:11434/api/generate"
    
    if speaker_name == "Unknown":
        prompt = "An unrecognized user just spoke to you. Warn them politely that they are an intruder, keeping it short."
    else:
        prompt = f"Welcome back user '{speaker_name}' with {confidence:.1f}% biometric confidence. Give them a unique, witty, and personalized welcome greeting in 2 sentences."
        
    data = {
        "model": "llama3",  
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        return response.json().get("response", "LLaMA failed to generate response.")
    except Exception:
        return f"[System Match]: Identified as {speaker_name}. (Ollama is not running to generate custom text)."

# 4. Function to identify the speaker from a fresh recording
def identify_voice(audio_path):
    if not audio_path:
        return "Please record or upload an audio file first.", ""
    if not voice_database:
        return "Database is empty! Please enroll at least one voice first.", ""
    
    try:
        unknown_embedding = get_audio_embedding(audio_path).reshape(1, -1)
        best_match = "Unknown"
        min_distance = 0.5  
        
        for name, stored_embedding in voice_database.items():
            stored_embedding = stored_embedding.reshape(1, -1)
            distance = cdist(unknown_embedding, stored_embedding, metric="cosine")[0,0]
            
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        confidence = (1 - min_distance) * 100
        llama_reply = ask_llama(best_match, confidence)
        status_report = f"Biometric Match: {best_match}\nDistance Score: {min_distance:.3f}"
        return status_report, llama_reply

    except Exception as e:
        return f"Error identifying voice: {str(e)}", ""

# 5. Build the Gradio Web Interface
with gr.Blocks(title="Voice Identity LLaMA Predictor") as demo:
    gr.Markdown("# 🎙️ Voice Identity & LLaMA Persona Predictor")
    gr.Markdown("Running stable SpeechBrain biometrics and local LLaMA processing loops.")
    
    with gr.Tab("1. Enroll Voice"):
        name_input = gr.Textbox(label="Enter Your Name")
        audio_enroll = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")
        enroll_btn = gr.Button("Save Voiceprint")
        enroll_output = gr.Textbox(label="Status")
        enroll_btn.click(enroll_voice, inputs=[name_input, audio_enroll], outputs=enroll_output)
        
    with gr.Tab("2. Predict Speaker"):
        audio_test = gr.Audio(sources=["microphone"], type="filepath", label="Record the mystery speaker")
        predict_btn = gr.Button("Who is speaking?")
        with gr.Row():
            bio_output = gr.Textbox(label="Biometric Metrics")
            llama_output = gr.Textbox(label="LLaMA Response Generator")
            
        predict_btn.click(identify_voice, inputs=[audio_test], outputs=[bio_output, llama_output])

if __name__ == "__main__":
    demo.launch()