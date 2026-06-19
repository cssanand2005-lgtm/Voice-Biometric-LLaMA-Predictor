Voice Identity \& LLaMA Persona Predictor



An end-to-end AI and Audio Data Science pipeline that extracts deep speaker embedding vectors to identify individuals and passes the identity context to a local LLaMA 3 engine to generate custom, witty personalized greetings.



🌟 Key Features



Parallel Biometric Auditing: Implemented utilizing pure Python array-processing frameworks for high-performance, low-latency voiceprint vector evaluations.

Deep Learning Voice Analytics: Leverages the pre-trained SpeechBrain ECAPA-TDNN neural architecture via PyTorch to extract high-fidelity, 192-dimensional speaker characteristics.

Live Audio Processing: Completely bypasses standard torchcodec and system FFmpeg binary constraints by intercepting vocal tracks directly via soundfile array handlers.

Local LLM Orchestration: Interfaces natively with a background Ollama daemon to ensure zero voice data or identity indicators ever leave the host machine.

Interactive UI Dashboard: Features a modern dual-tab Gradio frontend interface split seamlessly into dedicated Voice Enrollment and Identity Prediction modules.



🛠️ Project Architecture



app1.py - Serves the final interactive Gradio web app interface, orchestrates core biometric calculations, and handles the local LLaMA API loop generation.



🚀 How to Run Locally



Clone this repository.

Ensure you have your background Ollama service running (ollama run llama3).

Ensure you have your environment activated and run: pip install speechbrain gradio numpy scipy requests soundfile torchaudio

