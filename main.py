import os
import json
import subprocess
import requests
from datetime import datetime, timezone
from typing import Optional, List, Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

# ----------------- APP SETUP -----------------
app = FastAPI(title="Shortcut AI Video Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://rwgbfwexthxczemimldf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
BUCKET_NAME = "shortcut-videos"

# ----------------- PROMPTS -----------------

MOVIE_EXPLAINER_MASTER_PROMPT = """
You are a professional YouTube movie explanation editor.
Analyze the provided video and audio URLs/context and return a JSON object with:
1. "cuts": A list of start and end timestamps (format HH:MM:SS.mmm) capturing dynamic high-tension shots.
2. "subtitles": A list of short spoken phrases with start and end timestamps.

Output STRICTLY valid JSON matching this schema:
{
  "cuts": [
    {"start": "00:00:00.000", "end": "00:00:03.500"},
    {"start": "00:00:05.000", "end": "00:00:09.200"}
  ],
  "subtitles": [
    {"start": "00:00:00.500", "end": "00:00:03.000", "text": "THEY NEVER EXPECTED THIS"},
    {"start": "00:00:05.200", "end": "00:00:08.800", "text": "UNTIL IT WAS TOO LATE"}
  ]
}
"""

# ----------------- SCHEMAS -----------------

class CreateProjectRequest(BaseModel):
    user_id: str
    project_name: str
    template: str
    source_video_url: str
    audio_url: Optional[str] = None
    reference_video_url: Optional[str] = None
    optional_message: Optional[str] = None
    canvas_ratio: Optional[str] = "9:16"
    subtitle_font: Optional[str] = "Impact Yellow Glow"

class ChatMessageRequest(BaseModel):
    user_id: str
    project_name: str
    message: str

# ----------------- HELPERS -----------------

def download_file(url: str, dest_path: str):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=16384):
            f.write(chunk)

def get_user_record(user_id: str) -> dict:
    response = supabase.table("users").select("*").eq("id", user_id).execute()
    if not response.data or len(response.data) == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in database.")
    return response.data[0]

def update_user_projects(user_id: str, projects_data: dict):
    supabase.table("users").update({"projects": projects_data}).eq("id", user_id).execute()

def create_ass_file(subtitles: List[Dict[str, str]], ass_path: str):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,60,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,30,30,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for sub in subtitles:
            start_str = sub.get("start", "00:00:00.000")[:11]
            end_str = sub.get("end", "00:00:02.000")[:11]
            text = sub.get("text", "").upper()
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")

def run_ffmpeg_timeline_cut(cuts: List[Dict[str, str]], video_path: str, audio_path: Optional[str], ass_path: Optional[str], output_path: str, canvas_ratio: str = "9:16"):
    filter_complex = ""
    concat_inputs = ""
    crop_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" if canvas_ratio == "9:16" else "scale=1920:1080"

    for i, cut in enumerate(cuts):
        filter_complex += f"[0:v]trim=start='{cut['start']}':end='{cut['end']}',setpts=PTS-STARTPTS,{crop_filter}[v{i}]; "
        concat_inputs += f"[v{i}]"

    filter_complex += f"{concat_inputs}concat=n={len(cuts)}:v=1:a=0[stitched]; "

    if ass_path and os.path.exists(ass_path):
        filter_complex += f"[stitched]subtitles={ass_path}[outv]"
    else:
        filter_complex += f"[stitched]null[outv]"

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path, "-filter_complex", filter_complex, "-map", "[outv]", "-map", "1:a", "-shortest"])
    else:
        cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]"])

    cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", output_path])
    subprocess.run(cmd, check=True)

# ----------------- BACKGROUND WORKER -----------------

def process_video_edit_pipeline(user_id: str, project_name: str, template: str, video_url: str, audio_url: Optional[str], ref_url: Optional[str], canvas_ratio: str, subtitle_font: str):
    temp_dir = f"/tmp/{project_name}_{int(datetime.now().timestamp())}"
    os.makedirs(temp_dir, exist_ok=True)

    local_raw_video = os.path.join(temp_dir, "raw_video.mp4")
    local_audio = os.path.join(temp_dir, "narration.mp3") if audio_url else None
    local_ass = os.path.join(temp_dir, "subtitles.ass")
    local_output = os.path.join(temp_dir, "final_render.mp4")

    try:
        download_file(video_url, local_raw_video)
        if audio_url:
            download_file(audio_url, local_audio)

        # Call Gemini for scene selection and timestamps
        prompt_content = f"Video: {video_url}." + (f" Audio track: {audio_url}." if audio_url else "")
        ai_response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=MOVIE_EXPLAINER_MASTER_PROMPT,
                response_mime_type="application/json"
            )
        )
        parsed_data = json.loads(ai_response.text)
        cuts = parsed_data.get("cuts", [{"start": "00:00:00.000", "end": "00:00:10.000"}])
        subtitles = parsed_data.get("subtitles", [])

        if subtitles:
            create_ass_file(subtitles, local_ass)

        # Run FFmpeg compilation
        run_ffmpeg_timeline_cut(
            cuts=cuts,
            video_path=local_raw_video,
            audio_path=local_audio,
            ass_path=local_ass if os.path.exists(local_ass) else None,
            output_path=local_output,
            canvas_ratio=canvas_ratio
        )

        # Upload final render to Supabase Storage
        storage_path = f"renders/{user_id}/{project_name}.mp4"
        with open(local_output, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )

        render_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)

        # Update Project in DB
        user = get_user_record(user_id)
        projects = user.get("projects", {})
        if project_name in projects:
            projects[project_name]["status"] = "ready"
            projects[project_name]["render_output_url"] = render_url
            projects[project_name]["chat"].append({
                "sender": "Shot",
                "message": f"Render finished! Sliced {len(cuts)} cuts and overlaid animated captions.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ui_actions": {
                    "canvas_options": ["9:16", "16:9"],
                    "download_url": render_url
                }
            })
            update_user_projects(user_id, projects)

    except Exception as e:
        user = get_user_record(user_id)
        projects = user.get("projects", {})
        if project_name in projects:
            projects[project_name]["status"] = "failed"
            projects[project_name]["chat"].append({
                "sender": "Shot",
                "message": f"Rendering error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            update_user_projects(user_id, projects)
    finally:
        subprocess.run(["rm", "-rf", temp_dir])

# ----------------- ROUTES -----------------

@app.get("/")
def root():
    return {"status": "online", "engine": "Shortcut Video Engine"}

@app.post("/api/v1/projects/create")
async def create_project(req: CreateProjectRequest, background_tasks: BackgroundTasks):
    user = get_user_record(req.user_id)
    projects = user.get("projects", {}) or {}
    username = user.get("username", "User")

    chat_history = []
    if req.optional_message:
        chat_history.append({
            "sender": username,
            "message": req.optional_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    chat_history.append({
        "sender": "Shot",
        "message": f"Hey {username}! Assets received. Slicing scenes and styling vertical captions now...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    projects[req.project_name] = {
        "template": req.template,
        "status": "processing",
        "source_video_url": req.source_video_url,
        "audio_url": req.audio_url,
        "reference_video_url": req.reference_video_url,
        "render_output_url": None,
        "canvas_ratio": req.canvas_ratio,
        "subtitle_font": req.subtitle_font,
        "chat": chat_history
    }

    supabase.table("users").update({"projects": projects}).eq("id", req.user_id).execute()

    background_tasks.add_task(
        process_video_edit_pipeline,
        req.user_id,
        req.project_name,
        req.template,
        req.source_video_url,
        req.audio_url,
        req.reference_video_url,
        req.canvas_ratio,
        req.subtitle_font
    )

    return {"status": "success", "message": "Pipeline started", "project": projects[req.project_name]}

@app.get("/api/v1/projects/{user_id}/{project_name}")
async def get_project(user_id: str, project_name: str):
    user = get_user_record(user_id)
    projects = user.get("projects", {}) or {}
    if project_name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_name]
                  
