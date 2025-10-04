import os
import cv2
from typing import Dict, Any, List
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI
from django.conf import settings

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH2")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize YOLO + Groq client
model = YOLO(MODEL_PATH)
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def _condition_from_counts(potholes: int, cracks: int, erosion: int):
    pothole_score = potholes * 3
    crack_score = cracks * 2
    erosion_score = erosion * 4
    score = pothole_score + crack_score + erosion_score

    if potholes >= 6 and erosion >= 3:
        return "Critical", "Severe pothole and surface erosion detected.", score
    if potholes >= 6:
        return "Very Poor", "High number of potholes detected.", score
    if potholes >= 3 and erosion >= 2:
        return "Poor", "Potholes and erosion indicate poor quality.", score
    if erosion >= 4:
        return "Poor", "Significant surface erosion.", score
    if potholes >= 3 or cracks >= 6:
        return "Moderate", "Moderate road damage.", score
    if cracks >= 3:
        return "Fair", "Cracks found, not critical.", score
    return "Good", "Minor damage.", score


def _build_prompt(potholes: int, cracks: int, erosion: int, condition: str) -> str:
    detected = []
    if potholes > 0:
        detected.append(f"- Potholes: {potholes}")
    if cracks > 0:
        detected.append(f"- Cracks: {cracks}")
    if erosion > 0:
        detected.append(f"- Surface Erosion: {erosion}")

    body = "\n".join(detected) if detected else "No major damage detected."
    return f"""
    Detected road damage:
    {body}
    Overall road condition is rated as: {condition}.

    Write a short, professional report summarizing the road damage condition for municipal assessment. Use clear, non-technical language.
    """.strip()


def _ask_groq(prompt: str) -> str:
    """
    Query Groq LLM (Llama 3.3 70B) for report generation.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation failed ({e})."


def generate_road_damage_report(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline: YOLO detection + Groq LLM summary + detected image saving.
    Returns all information as a structured dictionary.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run YOLO
    results = model(image_path, imgsz=640)
    damage_counts = {"Pothole": 0, "Crack": 0, "Surface Erosion": 0}

    for cls_id in results[0].boxes.cls:
        label = model.names[int(cls_id)]
        if label in damage_counts:
            damage_counts[label] += 1

    potholes = damage_counts["Pothole"]
    cracks = damage_counts["Crack"]
    erosion = damage_counts["Surface Erosion"]
    print("hole")
    # Evaluate condition
    condition, explanation, score = _condition_from_counts(potholes, cracks, erosion)

    # Generate prompt and report
    prompt = _build_prompt(potholes, cracks, erosion, condition)
    report = _ask_groq(prompt)

    # Save annotated image
    detected_dir = os.path.join(settings.MEDIA_ROOT, "detected")
    os.makedirs(detected_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    detected_filename = f"{name_no_ext}_detected.jpg"
    detected_path = os.path.join(detected_dir, detected_filename)

    annotated_img = results[0].plot()
    cv2.imwrite(detected_path, annotated_img)

    # Build response dictionary
    return {
        "prompt": prompt.strip(),
        "report": report.strip(),
        "condition": condition,
        "explanation": explanation,
        "score": score,
        "damage_summary": {
            "potholes": potholes,
            "cracks": cracks,
            "surface_erosion": erosion,
        },
        "detected_filename": detected_filename,
    }
