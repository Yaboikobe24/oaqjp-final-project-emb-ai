"""Detect emotions using Watson NLP."""
import requests


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
    }
    keys = ("anger", "disgust", "fear", "joy", "sadness")
    response = requests.post(
        url, json={"raw_document": {"text": text_to_analyze}},
        headers=headers, timeout=30
    )
    if response.status_code == 400:
        return dict.fromkeys((*keys, "dominant_emotion"))
    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    result = {key: emotions[key] for key in keys}
    result["dominant_emotion"] = max(keys, key=result.get)
    return result
