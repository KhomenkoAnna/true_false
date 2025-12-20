import os
import onnx_asr

audio_folder = "/content/drive/MyDrive/audio_tests"
output_folder = "/content/drive/MyDrive/transcription_test"
os.makedirs(output_folder, exist_ok=True)
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]

model = onnx_asr.load_model("nemo-fastconformer-ru-rnnt")

transcriptions = model.recognize(audio_files)

for audio_path, transcription in zip(audio_files, transcriptions):
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder, file_name + ".txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

    print(f"Файлы сохранены: {output_path}")