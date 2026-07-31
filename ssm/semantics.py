# Expression labels remain ordered exactly as the dataset annotations.
SEVEN_CLASS_EMOTION_LABELS = [
    "happiness",
    "sadness",
    "neutral",
    "anger",
    "surprise",
    "disgust",
    "fear",
]

MAFW_EMOTION_LABELS = [
    "happiness",
    "sadness",
    "neutral",
    "anger",
    "surprise",
    "disgust",
    "fear",
    "contempt",
    "anxiety",
    "helplessness",
    "disappointment",
]

MAFW_EMOTION_DESCRIPTIONS = [
    "happiness, cheek raiser, lip corner puller",
    "sadness, inner brow raiser, brow lowerer, lip corner depressor",
    "neutral, relaxed facial muscles, no significant action units",
    "anger, brow lowerer, upper lid raiser, lip tightener, lips part",
    "surprise, inner brow raiser, outer brow raiser, upper lid raiser, jaw drop",
    "disgust, nose wrinkler, upper lip raiser, lip corner depressor",
    "fear, inner brow raiser, outer brow raiser, brow lowerer, upper lid raiser, jaw drop, lip stretcher",
    "contempt, unilateral lip corner tightener",
    "anxiety, inner brow raiser, brow lowerer, lip stretcher, lips part",
    "helplessness, inner brow raiser, brow lowerer, lip corner depressor, jaw drop",
    "disappointment, inner brow raiser, brow lowerer, slight lip corner depressor, lips part",
]

# AU identifiers and prompt descriptions are dataset-specific and ordered.
BP4D_AU_IDS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
BP4D_AU_DESCRIPTIONS = [
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
    "cheek raiser",
    "lid tightener",
    "upper lip raiser",
    "lip corner puller",
    "dimpler",
    "lip corner depressor",
    "chin raiser",
    "lip tightener",
    "lip presser",
]
BP4D_EMOTION_DESCRIPTIONS = [
    "happiness, cheek raiser, lip corner puller",
    "sadness, inner brow raiser, brow lowerer, lip corner depressor",
    "neutral, no active AUs",
    "anger, brow lowerer, lid tightener, lip tightener",
    "surprise, inner brow raiser, outer brow raiser",
    "disgust, upper lip raiser",
    "fear, inner brow raiser, outer brow raiser, brow lowerer",
]

DISFA_AU_IDS = [1, 2, 4, 6, 9, 12, 25, 26]
DISFA_AU_DESCRIPTIONS = [
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
    "cheek raiser",
    "nose wrinkler",
    "lip corner puller",
    "lips part",
    "jaw drop",
]
DISFA_EMOTION_DESCRIPTIONS = [
    "happiness, cheek raiser, lip corner puller",
    "sadness, inner brow raiser, brow lowerer",
    "neutral, relaxed facial muscles, straight mouth, smooth forehead, unremarkable eyebrows",
    "anger, brow lowerer",
    "surprise, inner brow raiser, outer brow raiser, jaw drop",
    "disgust, nose wrinkler",
    "fear, inner brow raiser, outer brow raiser, brow lowerer, jaw drop",
]


def get_emotion_labels(emotion_dataset):
    # Return a copy so callers cannot mutate the shared protocol constants.
    name = emotion_dataset.upper()
    if name in {"DFEW", "FERV39K"}:
        return SEVEN_CLASS_EMOTION_LABELS
    if name == "MAFW":
        return MAFW_EMOTION_LABELS
    raise ValueError(f"Unsupported emotion dataset: {emotion_dataset}")


def get_task_descriptions(emotion_dataset, au_dataset):
    # Pair the expression prompts with the selected AU vocabulary.
    emotion_name = emotion_dataset.upper()
    au_name = au_dataset.lower()
    if emotion_name == "MAFW":
        emotion_descriptions = MAFW_EMOTION_DESCRIPTIONS
    elif emotion_name in {"DFEW", "FERV39K"} and au_name == "bp4d":
        emotion_descriptions = BP4D_EMOTION_DESCRIPTIONS
    elif emotion_name in {"DFEW", "FERV39K"} and au_name == "disfa":
        emotion_descriptions = DISFA_EMOTION_DESCRIPTIONS
    else:
        raise ValueError(
            f"Unsupported dataset pair: {emotion_dataset} + {au_dataset}"
        )

    if au_name == "bp4d":
        return emotion_descriptions, BP4D_AU_DESCRIPTIONS
    if au_name == "disfa":
        return emotion_descriptions, DISFA_AU_DESCRIPTIONS
    raise ValueError(f"Unsupported AU dataset: {au_dataset}")
