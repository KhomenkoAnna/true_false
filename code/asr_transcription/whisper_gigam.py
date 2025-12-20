import os
import librosa
import soundfile as sf
import onnx_asr
import math

audio_folder = 'folder_path'
output_folder = 'folder_path'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

audio_files = [os.path.join(audio_folder, filename) for filename in os.listdir(audio_folder) if filename.endswith('.wav')]

model = onnx_asr.load_model("gigaam-v2-rnnt") #onnx-community/whisper-large-v3-turbo

for audio_file in audio_files:
    file_name = os.path.basename(audio_file).split('.')[0]

    audio, sr = librosa.load(audio_file, sr=16000)

    duration = librosa.get_duration(y=audio, sr=sr)

    chunk_duration = 30
    overlap_duration = 15
    step_duration = chunk_duration - overlap_duration
    num_chunks = math.ceil((duration - overlap_duration) / step_duration)

    full_transcription = ""

    for i in range(num_chunks):
        start_sample = i * step_duration * sr
        end_sample = min(start_sample + chunk_duration * sr, len(audio))
        audio_chunk = audio[start_sample:end_sample]

        chunk_file = f"/tmp/{file_name}_chunk_{i+1}.wav"
        sf.write(chunk_file, audio_chunk, sr)
        transcription = model.recognize([chunk_file])[0]
        full_transcription += transcription + "\n"

    output_txt_path = os.path.join(output_folder, f"{file_name}.txt")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription)

    print(f"Транскрипт для {file_name} сохранён в {output_txt_path}")
