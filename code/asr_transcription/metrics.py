import jiwer
import pandas as pd
import numpy as np

from jiwer import wer, cer, wil, wip


df = pd.read_excel(open('/content/Транскрипты.xlsx', 'rb'),
              sheet_name='Транскрипты')

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveSpecificWords(["эм", "аа", "ээ", "мм", "ам", "-"]),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

metrics = {
    "wer": [], "cer": [], "wil": [], "wip": []
}

for _, row in df.iterrows():
    gt = row["clean_transcription"]
    hyp = row["nemo-fastconformer-ru-rnnt"]

    # WER
    metrics["wer"].append(
        wer(
            gt, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    )

    # CER
    metrics["cer"].append(
        cer(
            gt, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    )

    # WIL
    metrics["wil"].append(
        wil(
            gt, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    )

    # WIP
    metrics["wip"].append(
        wip(
            gt, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
    )

mean_metrics = {k: np.mean(v) for k, v in metrics.items()}

print("Средние метрики по датасету:")
for name, value in mean_metrics.items():
    print(f"{name.upper()}: {value:.2%}")