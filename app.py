import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- IMPORTS ---
import torch
import cv2
import numpy as np
import base64
import io
import threading
from datetime import datetime
from PIL import Image, ImageOps

# Dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# Gen-AI
import google.generativeai as genai
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from dotenv import load_dotenv
load_dotenv()

# --- 1. GLOBAL STATE ---
job_state = {
    "is_running": False,
    "progress": 0,
    "status": "Ready",
    "result_image": None,
    "gemini_text": "",
    "filename": None
}

# --- 2. CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Gen-AI Engine running on: {device.upper()}")

# --- 3. LOAD MODELS ---
print("⏳ Loading Gen-AI Models...")
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/DreamShaper", 
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if device == "cuda": 
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    print("✅ Models Loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    pipe = None

# --- 4. UI LAYOUT ---
app = Dash(__name__, 
    external_stylesheets=[dbc.themes.CYBORG], 
    suppress_callback_exceptions=True, 
    title="Gen-AI Semantic Renderer"
)

app.layout = dbc.Container([
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True),

    # --- SAVE AS MODAL ---
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("💾 Save Image As"), close_button=True),
        dbc.ModalBody([
            html.P("Enter a name for your file:", className="text-muted"),
            dbc.Input(id="custom-filename", type="text", placeholder="e.g. Anime_Boy_Render", autoFocus=True),
            html.Small(".png will be added automatically", className="text-secondary")
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-cancel-save", className="ms-auto", color="secondary"),
            dbc.Button("⬇️ Save Now", id="btn-confirm-save", color="success", className="ms-2")
        ])
    ], id="download-modal", is_open=False, centered=True),

    dbc.Row([
        dbc.Col([
            html.H1("Gen-AI Semantic Sketch Renderer", className="text-center mt-4 mb-2", style={'fontWeight': 'bold', 'color': '#fff'}),
            
            html.Div([
                html.Span("Built with: ", className="text-muted me-2"),
                html.Span("Gemini 1.5 Flash", className="badge bg-primary me-2", style={'fontSize': '1em'}),
                html.Span("Stable Diffusion DreamShaper", className="badge bg-success me-2", style={'fontSize': '1em'}),
                html.Span("ControlNet", className="badge bg-info me-2", style={'fontSize': '1em'}),
                html.Span("Dash", className="badge bg-warning text-dark", style={'fontSize': '1em'}),
            ], className="text-center mb-4"),
            
            html.Hr(className="mb-5")
        ])
    ]),

    dbc.Row([
        # LEFT: CONTROLS
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Agent Control Panel", className="fw-bold text-white"),
                dbc.CardBody([
                    html.Label("Input Data (Sketch/Image)", className="text-info fw-bold"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([html.I(className="fas fa-cloud-upload-alt mr-2"), "Drag & Drop Image"]),
                        style={
                            'width': '100%', 'height': '80px', 'lineHeight': '80px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                            'textAlign': 'center', 'borderColor': '#0d6efd', 'backgroundColor': 'rgba(13, 110, 253, 0.1)'
                        },
                        multiple=False
                    ),
                    html.Hr(),
                    
                    html.Label("Semantic Context", className="text-success fw-bold"),
                    # --- CHANGED TO TEXTAREA ---
                    dbc.Textarea(
                        id="user-hint", 
                        placeholder="Describe the scene, oject or character details...'", 
                        style={'height': '100px'},
                        className="mb-2"
                    ),
                    
                    dbc.Checklist(
                        options=[{"label": "Auto-Invert (For White-on-Black images)", "value": "invert"}],
                        value=[],
                        id="invert-check",
                        switch=True,
                        className="mb-3 text-warning"
                    ),

                    html.Hr(),
                    dbc.Button("✨ Render Output", id="btn-run", color="primary", className="w-100 fw-bold shadow-lg", size="lg"),
                    
                    html.Div([
                        html.Label(id="progress-label", children="Agent Status: Idle", className="mt-3 text-warning"),
                        dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, color="warning", style={"height": "20px"}),
                    ], id="progress-container", style={"display": "none"}, className="mt-3"),
                ])
            ], className="shadow-lg border-0 h-100")
        ], width=4),

        # RIGHT: VIEWER
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Raw Input", className="text-center text-muted"),
                            html.Img(id='display-input', style={'width': '100%', 'borderRadius': '8px', 'border': '1px solid #444'})
                        ], width=6),
                        dbc.Col([
                            html.H6("Rendered Output", className="text-center text-success"),
                            html.Div(
                                html.Img(id='display-output', style={'width': '100%', 'borderRadius': '8px', 'border': '1px solid #444'}),
                                style={'minHeight': '200px'}
                            )
                        ], width=6),
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Span("🧠 Semantic Analysis (Gemini 1.5): ", className="fw-bold text-info"),
                        html.Span(id="gemini-prompt", style={"fontStyle": "italic", "color": "#ccc"})
                    ], className="p-3 bg-dark rounded border border-secondary"),
                    
                    dbc.Button("⬇️ Save High-Res Render", id="btn-open-modal", color="success", className="mt-3 w-100", disabled=True),
                    dcc.Download(id="download-component")
                ])
            ], className="shadow-lg border-0 h-100")
        ], width=8),
    ]),
    
    dcc.Store(id='stored-image-data'),
    dcc.Store(id='stored-filename'),

], fluid=True, className="pb-5")

# --- 5. LOGIC & WORKER THREAD ---

def process_input_image(pil_img, auto_invert=False):
    gray = pil_img.convert("L")
    stat =  np.array(gray).mean()
    if stat < 100 or auto_invert:
        pil_img = ImageOps.invert(pil_img.convert("RGB"))
    return pil_img

def get_canny_image(image):
    image = np.array(image)
    image = cv2.Canny(image, 50, 200) 
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def generate_prompt_with_gemini(image, user_hint=""):
    context = f"Context: {user_hint}" if user_hint else ""
    prompt_request = f"""
    Analyze this B&W line art. {context}
    Write a Stable Diffusion prompt to colorize it.
    - Style: 8k, masterpiece, vivid colors, volumetric lighting, high contrast.
    - Colors: Describe hair, eyes, skin, clothes, background.
    - Format: Comma-separated keywords.
    """
    try:
        response = gemini_model.generate_content([prompt_request, image])
        return response.text.strip()
    except:
        return f"{user_hint}, masterpiece, best quality, vivid colors, 8k"

def process_image_thread(pil_img, user_hint, should_invert):
    global job_state
    try:
        # 1. Pre-process
        job_state["status"] = "Preprocessing Input..."
        pil_img = process_input_image(pil_img, should_invert)
        
        # 2. Gemini Analysis
        job_state["status"] = "Gemini is Analyzing Semantic Context..."
        job_state["progress"] = 10
        gemini_prompt = generate_prompt_with_gemini(pil_img, user_hint)
        
        full_prompt = f"{gemini_prompt}, (masterpiece:1.2), (best quality), (vivid colors), (detailed light)"
        negative_prompt = "grayscale, monochrome, low quality, bad anatomy, blurry, washed out, pale, watermark, text"
        
        job_state["gemini_text"] = gemini_prompt
        
        # 3. Diffusion Generation
        job_state["status"] = "Latent Diffusion Rendering (Step 0/30)..."
        job_state["progress"] = 20
        canny_image = get_canny_image(pil_img)
        
        def sd_callback_on_step_end(pipe, step, timestep, callback_kwargs):
            p = 20 + int((step / 30) * 70)
            job_state["progress"] = p
            job_state["status"] = f"Latent Diffusion Rendering (Step {step}/30)..."
            return callback_kwargs

        output = pipe(
            full_prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=30,
            guidance_scale=8.5,
            controlnet_conditioning_scale=1.1,
            callback_on_step_end=sd_callback_on_step_end
        ).images[0]

        # 4. Finalize
        job_state["progress"] = 95
        job_state["status"] = "Finalizing Output..."
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"render_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        output.save(filepath)

        buff_out = io.BytesIO()
        output.save(buff_out, format="PNG")
        src_out = "data:image/png;base64," + base64.b64encode(buff_out.getvalue()).decode('utf-8')

        job_state["result_image"] = src_out
        job_state["filename"] = filename
        job_state["progress"] = 100
        job_state["status"] = "Render Complete!"
        job_state["is_running"] = False

    except Exception as e:
        print(f"Thread Error: {e}")
        job_state["status"] = f"Error: {str(e)}"
        job_state["is_running"] = False

# --- 6. CALLBACKS ---

@app.callback(
    Output('display-input', 'src'),
    Output('stored-image-data', 'data'),
    Input('upload-image', 'contents')
)
def update_input(contents):
    if not contents: return None, None
    return contents, contents

@app.callback(
    Output('progress-interval', 'disabled'),
    Output('progress-container', 'style'),
    Output('btn-run', 'disabled'),
    Input('btn-run', 'n_clicks'),
    State('stored-image-data', 'data'),
    State('user-hint', 'value'),
    State('invert-check', 'value'),
    prevent_initial_call=True
)
def start_generation(n, contents, user_hint, invert_val):
    if not contents or job_state["is_running"]: return True, {"display": "none"}, False
    
    job_state["is_running"] = True
    job_state["progress"] = 0
    job_state["status"] = "Initializing Agent..."
    job_state["result_image"] = None
    
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img).convert("RGB")
    pil_img = pil_img.resize((512, 512))

    should_invert = "invert" in invert_val

    t = threading.Thread(target=process_image_thread, args=(pil_img, user_hint, should_invert))
    t.daemon = True
    t.start()
    
    return False, {"display": "block"}, True

@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-label', 'children'),
    Output('display-output', 'src'),
    Output('gemini-prompt', 'children'),
    Output('stored-filename', 'data'),
    Output('btn-open-modal', 'disabled'),
    Output('progress-interval', 'disabled', allow_duplicate=True),
    Output('btn-run', 'disabled', allow_duplicate=True),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    if not job_state["is_running"]:
        if job_state["result_image"]:
            return (100, "Render Complete!", job_state["result_image"], job_state["gemini_text"], 
                    job_state["filename"], False, True, False)
        else:
            return (0, job_state["status"], None, "", None, True, True, False)
    return (job_state["progress"], job_state["status"], no_update, no_update, 
            no_update, True, False, True)

# --- MODAL LOGIC ---

@app.callback(
    Output("download-modal", "is_open"),
    Input("btn-open-modal", "n_clicks"),
    Input("btn-cancel-save", "n_clicks"),
    Input("btn-confirm-save", "n_clicks"),
    State("download-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n_open, n_cancel, n_confirm, is_open):
    if n_open or n_cancel or n_confirm:
        return not is_open
    return is_open

@app.callback(
    Output("download-component", "data"),
    Input("btn-confirm-save", "n_clicks"),
    State("custom-filename", "value"),
    State("stored-filename", "data"),
    prevent_initial_call=True
)
def download_file(n_clicks, custom_name, real_filename):
    if not n_clicks or not real_filename:
        return None
    
    final_name = real_filename
    if custom_name and custom_name.strip():
        clean_name = custom_name.strip()
        if not clean_name.lower().endswith(".png"):
            clean_name += ".png"
        final_name = clean_name
    
    file_path = os.path.join(OUTPUT_DIR, real_filename)
    return dcc.send_file(file_path, filename=final_name)

if __name__ == '__main__':
    app.run(debug=True)