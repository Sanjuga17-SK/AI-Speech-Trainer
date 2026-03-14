from agents.coordinator_agent import formatter_agent
from agents.tools.facial_expression_tool import analyze_facial_expressions
from agents.tools.voice_analysis_tool import analyze_voice_attributes
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Test the new direct orchestration flow
video_path = r"C:\Users\acer\AppData\Local\Temp\61885210_Recording 2026-03-13 215632.mp4"

print(f"\nStarting analysis for: {video_path}")
try:
    print("\n--- Running Tools Manually ---")
    facial_raw = analyze_facial_expressions.entrypoint(video_path)
    voice_raw = analyze_voice_attributes.entrypoint(video_path)
    
    print("\n--- Running Synthesis Agent (70B) ---")
    synthesis_prompt = f"""
    Analyze these public speaking results and provide a comprehensive coaching report in JSON format.
    
    FACIAL DATA:
    {facial_raw}
    
    VOICE DATA:
    {voice_raw}
    
    JSON SCHEMA REQUIRED:
    {{
      "facial_expression_response": "Qualitative summary",
      "voice_analysis_response": "Qualitative summary",
      "content_analysis_response": "Qualitative summary",
      "transcription": "FULL VERBATIM TRANSCRIPTION",
      "feedback_response": {{
        "scores": {{
          "content_organization": 4,
          "delivery_vocal_quality": 3,
          "body_language_eye_contact": 2,
          "audience_engagement": 3,
          "language_clarity": 4
        }},
        "total_score": 16,
        "interpretation": "Good",
        "feedback_summary": "Address the speaker"
      }},
      "strengths": ["list"],
      "weaknesses": ["list"],
      "suggestions": ["list"]
    }}
    """
    
    fmt = formatter_agent.run(synthesis_prompt)
    content = fmt.content
    
    if isinstance(content, str):
        content = json.loads(content[content.find('{'):content.rfind('}')+1])
    
    print("\n--- Final Synthesis Output ---")
    print(json.dumps(content, indent=2))
    
    print("\n--- Transcript Check ---")
    print(f"Transcript exists: {len(content.get('transcription', '')) > 20}")
    print(f"Sample: {content.get('transcription', '')[:50]}...")
    
    print("\nVerification: Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nCaught exception: {e}")
