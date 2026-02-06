import json
import numpy as np
import os
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from sklearn.metrics.pairwise import cosine_similarity

# ─── Конфигурация ───────────────────────────────────────────────────────────────
EMBEDDING_PB = "discogs-effnet-bs64-1.pb"
CLASSIFIER_PB = "genre_discogs400-discogs-effnet-1.pb"
CLASSIFIER_JSON = "genre_discogs400-discogs-effnet-1.json"
SAMPLE_RATE = 16000


def check_files():
    for f in [EMBEDDING_PB, CLASSIFIER_PB, CLASSIFIER_JSON]:
        if not os.path.exists(f):
            print(f"Ошибка: файл '{f}' не найден!")
            print("Скачайте модели здесь: https://essentia.upf.edu/models.html#discogs-effnet")
            exit(1)


def load_genre_tags():
    with open(CLASSIFIER_JSON, 'r') as f:
        metadata = json.load(f)
    tags = metadata['classes']
    print(f"Загружено {len(tags)} жанров/стилей Discogs-EffNet")
    print("Примеры:", ", ".join(tags[:6]), "...\n")
    return tags


def extract_embedding(audio_path):
    """Возвращает усреднённый 128/256-мерный эмбеддинг трека (Discogs-EffNet)"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    print(f"Загрузка {audio_path} ...")
    audio = MonoLoader(filename=audio_path, sampleRate=SAMPLE_RATE, resampleQuality=4)()
    print(f"→ длительность {len(audio) / SAMPLE_RATE:.1f} сек\n")

    print("Извлечение эмбеддингов Discogs-EffNet...")
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=EMBEDDING_PB,
        output="PartitionedCall:1"
    )
    frame_embeddings = embedding_model(audio)  # [frames, dim]

    # Усредняем по всем кадрам → один вектор на трек
    global_embedding = np.mean(frame_embeddings, axis=0)
    print(f"→ эмбеддинг получен, размерность = {global_embedding.shape[0]}\n")

    return global_embedding


def print_top_genres(audio_path, tags, top_n=12):
    """Опционально: топ-жанры по классификатору (для понимания стиля)"""
    audio = MonoLoader(filename=audio_path, sampleRate=SAMPLE_RATE, resampleQuality=4)()

    emb_model = TensorflowPredictEffnetDiscogs(graphFilename=EMBEDDING_PB, output="PartitionedCall:1")
    embeddings = emb_model(audio)

    classifier = TensorflowPredict2D(
        graphFilename=CLASSIFIER_PB,
        input="serving_default_model_Placeholder",
        output="PartitionedCall"
    )
    preds = classifier(embeddings)
    probs = np.mean(preds, axis=0)

    top_idx = np.argsort(probs)[-top_n:][::-1]

    print(f"Топ-{top_n} жанров для {os.path.basename(audio_path)}:")
    print("-" * 70)
    for i in top_idx:
        print(f"{tags[i]:.<48} {probs[i]:6.4f}  ({probs[i] * 100:5.1f}%)")
    print("-" * 70, "\n")


def compare_songs(path1, path2, tags, similarity_threshold=0.72, show_genres=True):
    print(f"\nСравнение двух треков:")
    print(f"  1. {path1}")
    print(f"  2. {path2}\n")

    vec1 = extract_embedding(path1)
    vec2 = extract_embedding(path2)

    # Косинусное сходство (чем ближе к 1 — тем более похожи)
    sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    print(f"Косинусное сходство эмбеддингов: {sim:.4f}")
    if sim >= similarity_threshold:
        print("→ Треки **очень похожи** по тембру, настроению, стилю")
    elif sim >= 0.60:
        print("→ Заметное сходство (однородный вайб)")
    elif sim >= 0.45:
        print("→ Среднее сходство (есть общие черты)")
    else:
        print("→ Сильно разные по звуку и характеру")

    if show_genres:
        print_top_genres(path1, tags)
        print_top_genres(path2, tags)


# ─── Запуск ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_files()
    tags = load_genre_tags()

    # ← Замените на свои файлы
    song1 = "How Do You Do! - Roxette.m4a"
    song2 = "Sleeping In My Car - Roxette.m4a"  # ← замените !

    compare_songs(song1, song2, tags, similarity_threshold=0.70, show_genres=True)

    # Можно сравнивать сколько угодно раз:
    # compare_songs("trackA.mp3", "trackB.mp3", tags)