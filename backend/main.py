from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agents.coordinator_agent import formatter_agent
from agents.tools.facial_expression_tool import analyze_facial_expressions
from agents.tools.voice_analysis_tool import analyze_voice_attributes
from agno.agent import RunOutput
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    video_url: str

def map_to_frontend_schema(data):
    """Robustly maps any LLM JSON output to the exact schema expected by the frontend."""
    if not isinstance(data, dict):
        return None
        
    def find_key(obj, target_key):
        if not isinstance(obj, dict): return None
        if target_key in obj: return obj[target_key]
        for k, v in obj.items():
            if isinstance(v, dict):
                res = find_key(v, target_key)
                if res is not None: return res
        return None

    scores_raw = find_key(data, "scores") or {}
    scores = {
        "content_organization": scores_raw.get("content_organization", 0),
        "delivery_vocal_quality": scores_raw.get("delivery_vocal_quality", 0),
        "body_language_eye_contact": scores_raw.get("body_language_eye_contact", 0),
        "audience_engagement": scores_raw.get("audience_engagement", 0),
        "language_clarity": scores_raw.get("language_clarity", 0),
    }

    # Extract transcription explicitly
    transcription = data.get("transcription") or find_key(data, "transcription") or find_key(data, "transcript")

    mapped = {
        "facial_expression_response": data.get("facial_expression_response") or str(find_key(data, "facial_expression") or "Analysis complete"),
        "voice_analysis_response": data.get("voice_analysis_response") or str(find_key(data, "voice_analysis") or "Analysis complete"),
        "content_analysis_response": data.get("content_analysis_response") or str(find_key(data, "content_analysis") or "Analysis complete"),
        "transcription": transcription or "Transcript not found.",
        "feedback_response": {
            "scores": scores,
            "total_score": data.get("total_score") or find_key(data, "total_score") or sum(scores.values()),
            "interpretation": data.get("interpretation") or find_key(data, "interpretation") or "Scale result",
            "feedback_summary": data.get("feedback_summary") or find_key(data, "feedback_summary") or "Good effort!"
        },
        "strengths": data.get("strengths") or find_key(data, "strengths") or [],
        "weaknesses": data.get("weaknesses") or find_key(data, "weaknesses") or [],
        "suggestions": data.get("suggestions") or find_key(data, "suggestions") or []
    }
    return mapped

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    video_url = request.video_url
    
    try:
        # Step 1: Manual Tool Execution (Guaranteed)
        print(f"Orchestrating tools for: {video_url}")
        facial_data = analyze_facial_expressions.entrypoint(video_url)
        voice_data = analyze_voice_attributes.entrypoint(video_url)
        
        # Step 2: Qualitative Synthesis (70B Agent)
        synthesis_prompt = f"""
        Analyze these public speaking results and provide a comprehensive coaching report in JSON format.
        
        FACIAL DATA:
        {facial_data}
        
        VOICE DATA:
        {voice_data}
        
        JSON SCHEMA REQUIRED:
        {{
          "facial_expression_response": "Qualitative summary of emotions/eye contact",
          "voice_analysis_response": "Qualitative summary of voice metrics",
          "content_analysis_response": "Qualitative summary of speech content",
          "transcription": "FULL VERBATIM TRANSCRIPTION FROM DATA",
          "feedback_response": {{
            "scores": {{
              "content_organization": [1-5],
              "delivery_vocal_quality": [1-5],
              "body_language_eye_contact": [1-5],
              "audience_engagement": [1-5],
              "language_clarity": [1-5]
            }},
            "total_score": [sum],
            "interpretation": "[Scale]",
            "feedback_summary": "[Direct coaching address to the speaker]"
          }},
          "strengths": ["...", "..."],
          "weaknesses": ["...", "..."],
          "suggestions": ["...", "..."]
        }}
        """
        
        # Use the formatter agent (70B) for synthesis
        response: RunOutput = formatter_agent.run(synthesis_prompt)
        content = response.content
        
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}')
            content = json.loads(content[start:end+1])
            
        final_data = map_to_frontend_schema(content)
        return JSONResponse(content=jsonable_encoder(final_data))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)