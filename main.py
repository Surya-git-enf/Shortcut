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
from faster_whisper import WhisperModel

# ----------------- APP & CLIENT SETUP -----------------
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
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

BUCKET_NAME = "shortcut-videos"

# ----------------- PROMPTS -----------------

MOVIE_EXPLAINER_MASTER_PROMPT = """
You are a professional YouTube movie explanation editor specializing in high-retention, cinematic storytelling. 
Your task is to analyze the provided movie footage and convert it into a highly engaging, suspense-driven video cut synced with the audio/script.

OBJECTIVE:
Transform the raw movie into a cinematic video that feels immersive, emotionally intense, and structured for maximum viewer retention.

RULES:
1. Pacing & Hooks: The first 15-20 seconds MUST use fast cuts (0.8s - 1.5s per shot) to build immediate tension.
2. Chronological & Narrative Sync: Mirror the storytelling rhythm and match the visual action directly with the scene described in the voiceover line.
3. No Dead Air: Remove static shots, unnecessary pauses, and filler dialogue.
4. Output Schema: You MUST respond ONLY with a valid JSON array of timestamps:
[
  {"start": "00:00:02.000", "end": "00:00:04.500", "description": "High tension hook shot"},
  {"start": "00:00:08.100", "end": "00:00:10.200", "description": "Turning point scene"}
]
"""

REFERENCE_VIDEO_MATCHER_PROMPT = """
You are an expert AI video editor. Edit the raw video using the reference video as the primary creative reference.

INPUTS & RULES:
1. Pacing & Cuts: Analyze the reference video's pacing, cut frequency, speed ramps, and shot transitions, then apply that rhythm to the raw footage.
2. Audio Sync: Match major visual impacts and transitions with the audio's beats and moments.
3. Quality Requirements: Select the strongest, most dynamic moments of the raw footage; eliminate dead time.
4. Output Schema: Return ONLY a valid JSON array of timestamps:
[
  {"start": "00:00:01.000", "end": "00:00:03.200", "description": "Fast action cut"},
  {"start": "00:00:05.500", "end": "00:00:08.000", "description": "Impact scene"}
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
    canvas_ratio: Optional[str] = "9:16"  # "9:16" or "16:9"
    subtitle_font: Optional[str] = "Impact Yellow Glow"

class ChatMessageRequest(BaseModel):
    user_id: str
    project_name: str
    message: str

# ----------------- HELPER FUNCTIONS -----------------

def download_file(url: str, dest_path: str):
    """Downloads a file from a public URL to local storage."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def get_user_record(user_id: str) -> dict:
    response = supabase.table("users").select("*").eq("id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    return response.data

def update_user_projects(user_id: str, projects_data: dict):
    supabase.table("users").update({"projects": projects_data}).eq("id", user_id).execute()

def generate_ass_subtitles(audio_path: str, ass_path: str, font_style: str = "Impact Yellow Glow"):
    """Transcribes audio with word-level timestamps and formats an .ass subtitle file."""
    segments, _ = whisper_model.transcribe(audio_path, word_timestamps=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,65,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,0,2,40,40,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for segment in segments:
            start_m, start_s = divmod(segment.start, 60)
            start_h, start_m = divmod(start_m, 60)
            end_m, end_s = divmod(segment.end, 60)
            end_h, end_m = divmod(end_m, 60)

            start_str = f"{int(start_h):01d}:{int(start_m):02d}:{start_s:05.2f}"
            end_str = f"{int(end_h):01d}:{int(end_m):02d}:{end_s:05.2f}"

            text = segment.text.strip().upper()
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n")

def run_ffmpeg_timeline_cut(
    cuts: List[Dict[str, str]],
    video_path: str,
    audio_path: Optional[str],
    ass_path: Optional[str],
    output_path: str,
    canvas_ratio: str = "9:16"
):
    """Trims selected scenes, stacks filters for 9:16 smart cropping, and burns subtitles."""
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

    if audio_path:
        cmd.extend([
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "1:a",
            "-shortest"
        ])
    else:
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]"
        ])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        output_path
    ])

    subprocess.run(cmd, check=True)

# ----------------- BACKGROUND EDITING WORKER -----------------

def process_video_edit_pipeline(
    user_id: str,
    project_name: str,
    template: str,
    source_video_url: str,
    audio_url: Optional[str],
    ref_url: Optional[str],
    canvas_ratio: str,
    subtitle_font: str
):
    temp_dir = f"/tmp/{project_name}_{int(datetime.now().timestamp())}"
    os.makedirs(temp_dir, exist_ok=True)

    local_raw_video = os.path.join(temp_dir, "raw_video.mp4")
    local_audio = os.path.join(temp_dir, "narration.mp3") if audio_url else None
    local_ass = os.path.join(temp_dir, "subtitles.ass")
    local_output = os.path.join(temp_dir, "final_render.mp4")

    try:
        # 1. Download Input Media
        download_file(source_video_url, local_raw_video)
        if audio_url:
            download_file(audio_url, local_audio)

        # 2. Select Prompt & Call Gemini
        system_instruction = MOVIE_EXPLAINER_MASTER_PROMPT if template == "movie_explanation" else REFERENCE_VIDEO_MATCHER_PROMPT
        prompt_content = f"Analyze source video at: {source_video_url}. "
        if audio_url:
            prompt_content += f"Match timing with audio: {audio_url}. "
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

        # 3. Generate Word-Level Subtitles
        if local_audio and os.path.exists(local_audio):
            generate_ass_subtitles(local_audio, local_ass, subtitle_font)

        # 4. Render Final Video via FFmpeg
        run_ffmpeg_timeline_cut(
            cuts=cuts,
            video_path=local_raw_video,
            audio_path=local_audio,
            ass_path=local_ass if (local_audio and os.path.exists(local_ass)) else None,
            output_path=local_output,
            canvas_ratio=canvas_ratio
        )

        # 5. Upload Rendered MP4 to Supabase Storage
        storage_path = f"renders/{user_id}/{project_name}.mp4"
        with open(local_output, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "video/mp4", "upsert": "true"}
            )

        render_url = supabase.storage.from_(BUCKET_NAME).get_public_url(storage_path)

        # 6. Update Project Record & Chat History in Supabase
        user = get_user_record(user_id)
        projects = user.get("projects", {})

        if project_name in projects:
            projects[project_name]["status"] = "ready"
            projects[project_name]["render_output_url"] = render_url
            projects[project_name]["chat"].append({
                "sender": "Shot",
                "message": f"I've completed your professional edit with {len(cuts)} synchronized cuts and burned animated subtitles! Check out the preview.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ui_actions": {
                    "canvas_options": ["9:16", "16:9"],
                    "subtitle_fonts": ["Impact Yellow Glow", "Bold White", "Clean Sans"],
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
        # Cleanup temporary files
        subprocess.run(["rm", "-rf", temp_dir])

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
        "message": f"Hey {username}! I've received your video assets. I am now analyzing the footage, syncing the narration, and cutting the timeline...",
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

    # Deduct credits & update DB
    supabase.table("users").update({
        "projects": projects,
        "credits": max(0, user.get("credits", 100) - 10)
    }).eq("id", req.user_id).execute()

    # Trigger async video generation pipeline
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

    return {
        "status": "success",
        "message": "Project created. AI editing started.",
        "project": projects[req.project_name]
    }

@app.post("/api/v1/projects/chat")
async def chat_with_shot(req: ChatMessageRequest):
    user = get_user_record(req.user_id)
    projects = user.get("projects", {})
    username = user.get("username", "User")

    if req.project_name not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[req.project_name]

    # Save User message
    project["chat"].append({
        "sender": username,
        "message": req.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # AI conversational response
    ai_response = gemini_client.models.generate_content(
        model='gemini-1.5-flash',
        contents=f"User request: '{req.message}'. You are Shot, an expert AI video copilot inside Shortcut. Acknowledge the request conversationally and summarize the changes."
    )

    project["chat"].append({
        "sender": "Shot",
        "message": ai_response.text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ui_actions": {
            "canvas_options": ["9:16", "16:9"],
            "subtitle_fonts": ["Impact Yellow Glow", "Clean Sans", "Bold White"]
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
  \
