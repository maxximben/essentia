import json
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
from sklearn.metrics.pairwise import cosine_similarity

# ─── Конфигурация ────────────────────────────────────────────────
MODEL_GRAPH = "models/msd-musicnn-1.pb"
METADATA_FILE = "msd-musicnn-1.json"
SAMPLE_RATE = 16000


def load_tags(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['classes']


def extract_tag_vector(audio_path, graph_filename, output_layer="model/Sigmoid"):
    """
    Извлекает 50-мерный вектор усреднённых вероятностей тегов для трека
    Возвращает: numpy массив shape=(50,)
    """
    # Загрузка аудио
    audio = MonoLoader(filename=audio_path, sampleRate=SAMPLE_RATE, resampleQuality=4)()

    # Предсказание модели по кадрам
    framewise = TensorflowPredictMusiCNN(
        graphFilename=graph_filename,
        output=output_layer
    )(audio)

    # Усредняем по времени → глобальный вектор
    vector = np.mean(framewise, axis=0)
    return vector


def print_top_tags(vector, tags, top_n=10):
    """Выводит топ-N тегов с вероятностями"""
    top_indices = np.argsort(vector)[-top_n:][::-1]
    print(f"  Топ-{top_n} тегов:")
    print("-" * 50)
    for idx in top_indices:
        tag = tags[idx]
        prob = vector[idx]
        print(f"  {tag:.<25} {prob:>6.3f}  ({prob * 100:5.1f}%)")
    print("-" * 50)


def compare_two_songs(path1, path2, tags, threshold=0.70):
    """Сравнивает две песни и выводит косинусное сходство"""
    print(f"\nСравниваем:")
    print(f"  1. {path1}")
    print(f"  2. {path2}\n")

    vec1 = extract_tag_vector(path1, MODEL_GRAPH)
    vec2 = extract_tag_vector(path2, MODEL_GRAPH)

    # Косинусное сходство (значение от -1 до 1, но в данном случае почти всегда > 0)
    sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    print(f"Косинусное сходство: {sim:.4f}")
    if sim >= threshold:
        print("→ Песни довольно похожи по настроению/жанру/тембру")
    elif sim >= 0.50:
        print("→ Есть заметное сходство")
    else:
        print("→ Стили заметно различаются")

    print("\nТоп-теги первой песни:")
    print_top_tags(vec1, tags)

    print("\nТоп-теги второй песни:")
    print_top_tags(vec2, tags)


# ─── Запуск ───────────────────────────────────────────────────────
if __name__ == "__main__":
    tags = load_tags(METADATA_FILE)
    print(f"Модель загружена. Количество тегов: {len(tags)}\n")
    print("Примеры тегов:", ", ".join(tags[:8]), "...\n")

    # ← Здесь укажите пути к вашим файлам
    song1 = "How Do You Do! - Roxette.m4a"
    song2 = "Sleeping In My Car - Roxette.m4a"  # ← замените !

    # Можно сравнивать несколько пар подряд
    compare_two_songs(song1, song2, tags, threshold=0.68)

    # Пример с другой парой:
    # compare_two_songs("trackA.mp3", "trackB.mp3", tags)