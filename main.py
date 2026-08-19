import os
import json
import subprocess
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

# ----------------- INITIALIZATION -----------------
app = FastAPI(title="Shortcut AI Video Engine")

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

BUCKET_NAME = "shortcut-videos"

# ----------------- PROMPTS -----------------

MOVIE_EXPLAINER_EDITOR_PROMPT = """
You are a professional AI YouTube Movie Explanation Video Editor. Your task is to analyze the provided raw movie footage and synchronize its visual cuts directly to match the provided voiceover/narration beats.

OBJECTIVE:
Transform the raw movie footage into a cinematic, high-retention video edit that precisely visualizes the spoken narration.

RULES:
1. Pacing & Hooks: The first 15-20 seconds must use fast cuts (0.8s - 1.5s per shot) to build immediate tension.
2. Narrative Sync: Match the visual action directly with the scene described in the voiceover line.
3. No Dead Air: Trim pauses, filler dialogue, and static shots.
4. Output Schema: You MUST respond ONLY with a JSON array containing timestamps:
[
  {"start": "00:09:20", "end": "00:09:23", "description": "Character enters stall"},
  {"start": "00:10:45", "end": "00:10:48", "description": "Close up struggle"}
]
"""

REFERENCE_VIDEO_MATCHER_PROMPT = """
You are an expert AI Video Editor. Edit the raw footage using the reference video as the visual rhythm and structure template.

RULES:
1. Pacing & Energy: Match the cut frequency, speed ramps, and shot transitions of the reference video.
2. Synchronize cuts to the selected audio track duration.
3. Select the most dynamic moments from the raw footage.
4. Output Schema: Return strictly a JSON array of cuts:
[
  {"start": "00:01:10", "end": "00:01:12", "description": "Fast action cut"},
  {"start": "00:02:40", "end": "00:02:43", "description": "Impact scene"}
]
"""

# ----------------- SCHEMAS -----------------

class CreateProjectRequest(BaseModel):
    user_id: str
    project_name: str
    template: str  # "movie_explanation" | "reference_match" | "new"
    source_video_url: str
    audio_url: Optional[str] = None
    reference_video_url: Optional[str] = None
    optional_message: Optional[str] = None

class ChatMessageRequest(BaseModel):
    user_id: str
    project_name: str
    message: str

# ----------------- HELPERS -----------------

def get_user_record(user_id: str):
    response = supabase.table("users").select("*").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    return response.data

def update_user_projects(user_id: str, projects_data: dict):
    supabase.table("users").update({"projects": projects_data}).eq("id", user_id).execute()

def run_ffmpeg_timeline_cut(cuts: list, video_url: str, audio_url: Optional[str], output_path: str, canvas_ratio: str = "9:16"):
    """Slices video using FFmpeg and stitches it to the audio duration."""
    filter_complex = ""
    concat_inputs = ""
    
    crop_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" if canvas_ratio == "9:16" else "scale=1920:1080"
    
    for i, cut in enumerate(cuts):
        filter_complex += f"[0:v]trim=start='{cut['start']}':end='{cut['end']}',setpts=PTS-STARTPTS,{crop_filter}[v{i}]; "
        concat_inputs += f"[v{i}]"

    filter_complex += f"{concat_inputs}concat=n={len(cuts)}:v=1:a=0[outv]"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_url,
    ]
    
    if audio_url:
        cmd.extend(["-i", audio_url, "-filter_complex", filter_complex, "-map", "[outv]", "-map", "1:a", "-shortest"])
    else:
        cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]"])
        
    cmd.extend(["-c:v", "libx264", "-c:a", "aac", output_path])
    subprocess.run(cmd, check=True)

# ----------------- BACKGROUND WORKERS -----------------

def process_video_edit_pipeline(user_id: str, project_name: str, template: str, video_url: str, audio_url: Optional[str], ref_url: Optional[str]):
    try:
        # 1. Select Prompt
        system_instruction = MOVIE_EXPLAINER_EDITOR_PROMPT if template == "movie_explanation" else REFERENCE_VIDEO_MATCHER_PROMPT
        
        # 2. Call Gemini 1.5
        prompt_content = f"Analyze source video: {video_url}. "
        if audio_url:
            prompt_content += f"Match narration/audio: {audio_url}. "
        if ref_url:
            prompt_content += f"Mirror reference pacing from: {ref_url}. "

        ai_response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        
        cuts = json.loads(ai_response.text)
        
        # 3. Render MP4 Locally
        local_output = f"/tmp/{project_name}_render.mp4"
        run_ffmpeg_timeline_cut(cuts, video_url, audio_url, local_output)
        
        # 4. Upload Render Directly to Supabase Storage
        storage_path = f"renders/{user_id}/{project_name}.mp4"
        with open(local_output, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=storage_path,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )
        
        # Get Public URL from Supabase
        render_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)
        
        # 5. Update Supabase Database
        user = get_user_record(user_id)
        projects = user.get("projects", {})
        
        if project_name in projects:
            projects[project_name]["status"] = "ready"
            projects[project_name]["render_output_url"] = render_url
            projects[project_name]["chat"].append({
                "sender": "Shot",
                "message": f"I've completed your video edit with {len(cuts)} synchronized cuts! Check out the preview.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ui_actions": {
                    "canvas_options": ["9:16", "16:9"],
                    "subtitle_fonts": ["Impact Yellow Glow", "Bold White"],
                    "download_url": render_url
                }
            })
            update_user_projects(user_id, projects)
            
        # Clean local temp file
        if os.path.exists(local_output):
            os.remove(local_output)

    except Exception as e:
        user = get_user_record(user_id)
        projects = user.get("projects", {})
        if project_name in projects:
            projects[project_name]["status"] = "failed"
            projects[project_name]["chat"].append({
                "sender": "Shot",
                "message": f"Error rendering your video: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            update_user_projects(user_id, projects)

# ----------------- API ROUTES -----------------

@app.post("/api/v1/projects/create")
async def create_project(req: CreateProjectRequest, background_tasks: BackgroundTasks):
    user = get_user_record(req.user_id)
    projects = user.get("projects", {})
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
        "message": f"Hey {username}! I've received your assets. I am now watching the video and preparing the initial cut...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    projects[req.project_name] = {
        "template": req.template,
        "status": "processing",
        "source_video_url": req.source_video_url,
        "audio_url": req.audio_url,
        "reference_video_url": req.reference_video_url,
        "render_output_url": None,
        "canvas_ratio": "9:16",
        "chat": chat_history
    }

    # Deduct credits & save
    supabase.table("users").update({
        "projects": projects,
        "credits": max(0, user.get("credits", 100) - 10)
    }).eq("id", req.user_id).execute()

    # Trigger async edit task
    background_tasks.add_task(
        process_video_edit_pipeline,
        req.user_id,
        req.project_name,
        req.template,
        req.source_video_url,
        req.audio_url,
        req.reference_video_url
    )

    return {"status": "success", "message": "Project created. Processing started.", "project": projects[req.project_name]}


@app.post("/api/v1/projects/chat")
async def chat_with_shot(req: ChatMessageRequest):
    user = get_user_record(req.user_id)
    projects = user.get("projects", {})
    username = user.get("username", "User")

    if req.project_name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_name]

    # Append User Message
    project["chat"].append({
        "sender": username,
        "message": req.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # Generate Shot AI Response via Gemini
    ai_response = gemini_client.models.generate_content(
        model='gemini-1.5-flash',
        contents=f"User says: '{req.message}'. You are Shot, an AI video copilot inside Shortcut. Respond conversationally and specify what changes you are applying."
    )

    # Append AI Response
    project["chat"].append({
        "sender": "Shot",
        "message": ai_response.text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ui_actions": {
            "canvas_options": ["9:16", "16:9"],
            "subtitle_fonts": ["Impact Yellow Glow", "Clean Sans"]
        }
    })

    update_user_projects(req.user_id, projects)
    return {"status": "success", "chat": project["chat"]}


@app.get("/api/v1/projects/{user_id}/{project_name}")
async def get_project_status(user_id: str, project_name: str):
    user = get_user_record(user_id)
    projects = user.get("projects", {})
    if project_name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_name]
    
