"""
Enhanced Text-to-Speech Processor with Segment-Specific Instructions
-------------------------------------------
A modern TTS application using OpenAI's GPT-4o-mini-tts model 
that applies different style instructions to different segment types
(narrative, dialog, code) for more natural and engaging narration.

Supports all 11 available voices with customizable instructions per segment type.
"""

import re
import os
import json
import openai
import threading
import subprocess
import tempfile
from docx import Document
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import time

# Configuration
# The 11 voices available in the GPT-4o-mini-tts model
VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral", 
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse"
]

# Voice roles
VOICE_ROLES = {
    "narrator": "onyx",     # Deep, authoritative voice for narration
    "dialog": "nova",       # Warm, friendly voice for dialog
    "computer": "echo"      # Technical voice for code/computer text
}

# Common style instructions per segment type
STYLE_PRESETS = {
    "narrator": [
        "Default (no style instruction)",
        "Speak in a calm and measured tone, with thoughtful pauses",
        "Speak with an authoritative tone, like narrating a documentary",
        "Speak in a mysterious and atmospheric way",
        "Speak with a sense of wonder and intrigue",
        "Custom style..."
    ],
    "dialog": [
        "Default (no style instruction)",
        "Speak with natural conversational inflection",
        "Speak with emotional expressiveness",
        "Speak with subtle character distinctions",
        "Speak as if telling a story to a friend",
        "Custom style..."
    ],
    "computer": [
        "Default (no style instruction)",
        "Speak in a technical and precise manner",
        "Speak with a slightly computerized tone",
        "Speak with clear articulation of technical terms",
        "Speak at a slightly slower pace for code segments",
        "Custom style..."
    ]
}

# Default settings
DEFAULT_MODEL = "gpt-4o-mini-tts"  # The new model
MAX_CHUNK_SIZE = 4000  # Characters per chunk (limit for processing)

class TextAnalyzer:
    """Analyzes text to determine section types and appropriate voices"""
    
    def identify_paragraph_type(self, paragraph):
        """Determine the type of a paragraph based on content patterns"""
        
        # Skip empty paragraphs
        if not paragraph.strip():
            return "narrator"
            
        # Check for code blocks or terminal output
        if (paragraph.strip().startswith('```') or 
            paragraph.strip().startswith('>') or 
            paragraph.strip().startswith('File:') or
            re.search(r'Filename:|Log:|Author:|Last Modified', paragraph)):
            return "computer"
            
        # Check for dialogue-heavy paragraphs
        dialogue_matches = re.findall(r'"[^"]*"', paragraph)
        if dialogue_matches and len(''.join(dialogue_matches)) > len(paragraph) * 0.3:
            return "dialog"
            
        # Default to narrator
        return "narrator"
    
    def analyze_text(self, full_text):
        """Split text into segments and assign voice types"""
        segments = []
        
        # Split by paragraphs (respecting blank lines)
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            paragraph_type = self.identify_paragraph_type(paragraph)
            
            # For longer paragraphs, check if we need to further split
            if len(paragraph) > 1000:
                # Split by sentences for long paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > 1000:
                        if current_chunk:
                            segments.append((paragraph_type, current_chunk.strip()))
                            current_chunk = sentence
                        else:
                            segments.append((paragraph_type, sentence.strip()))
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                if current_chunk:
                    segments.append((paragraph_type, current_chunk.strip()))
            else:
                segments.append((paragraph_type, paragraph.strip()))
        
        return segments

class TextFormatter:
    """Enhances text for better TTS performance"""
    
    def format_for_tts(self, text_type, text):
        """Format text for better narration based on text type"""
        
        if text_type == "computer":
            # Format code or terminal text for better narration
            if text.strip().startswith('```') and '```' in text[3:]:
                # Code block formatting
                code_text = text.strip('`').strip()
                # Add spacing to improve code readability when spoken
                formatted_text = re.sub(r'([=<>\(\)\[\]\{\}])', r' \1 ', code_text)
                return formatted_text
            elif text.strip().startswith('>'):
                # Terminal output
                return text.replace('>', '').strip()
            else:
                return text
            
        elif text_type == "dialog":
            # Enhance dialog with better flow
            # Ensure proper spacing around quotes
            text = re.sub(r'(\w)"', r'\1" ', text)
            text = re.sub(r'"(\w)', r' "\1', text)
            return text
            
        # Default case - just return cleaned text
        return text.strip()

class EnhancedTTSConverter:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or "OpenAI-API-Key-Here"
        openai.api_key = self.api_key
        self.analyzer = TextAnalyzer()
        self.formatter = TextFormatter()
        
    def get_text_from_file(self, file_path):
        """Extract text from supported file formats"""
        ext = os.path.splitext(file_path)[-1].lower()
        
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".docx":
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            raise ValueError("Unsupported file format. Use .txt or .docx")
    
    def process_text(self, text, voice_roles):
        """Process text into voice-assigned segments"""
        segments = self.analyzer.analyze_text(text)
        enhanced_segments = []
        
        for text_type, content in segments:
            formatted_text = self.formatter.format_for_tts(text_type, content)
            
            # Map text type to voice based on user settings
            voice = voice_roles.get(text_type, voice_roles["narrator"])
                
            enhanced_segments.append((text_type, voice, formatted_text))
            
        return enhanced_segments
        
    def text_to_speech(self, text, filename, voice, instructions="", model=DEFAULT_MODEL):
        """Convert text to speech using OpenAI's TTS API"""
        # Make the API call
        try:
            # Prepare request parameters
            params = {
                "model": model,
                "voice": voice,
                "input": text
            }
            
            # Add instructions if provided
            if instructions and instructions != "Default (no style instruction)":
                params["instructions"] = instructions
            
            response = openai.audio.speech.create(**params)
            
            with open(filename, "wb") as f:
                f.write(response.content)
                
            return True
            
        except Exception as e:
            print(f"Error in TTS API call: {str(e)}")
            raise e
        
    def convert_with_chunking(self, text, filename, voice, instructions="", model=DEFAULT_MODEL):
        """Convert text to speech with chunking for longer texts"""
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Create a temporary directory for chunks
        temp_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        name, ext = os.path.splitext(base_name)
        
        chunk_files = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"{name}_part{i}{ext}")
            chunk_files.append(chunk_file)
            
            self.text_to_speech(chunk, chunk_file, voice, instructions, model)
        
        # If we only have one chunk, just rename it
        if len(chunk_files) == 1:
            os.rename(chunk_files[0], filename)
            return
            
        # Combine audio files if multiple chunks
        self._combine_audio_files(chunk_files, filename)
        
        # Clean up temporary files
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
                
    def _split_text(self, text):
        """Split text into chunks of appropriate size for the API"""
        # Use a smaller limit than the max to be safe
        CHUNK_SIZE = MAX_CHUNK_SIZE
        
        if len(text) <= CHUNK_SIZE:
            return [text]
            
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds the chunk size
            if len(current_chunk) + len(paragraph) + 1 > CHUNK_SIZE:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If paragraph itself is too long, split it
                if len(paragraph) > CHUNK_SIZE:
                    # Split by sentences (simple approximation)
                    sentences = paragraph.replace('. ', '.|').split('|')
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > CHUNK_SIZE:
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = ""
                            
                            # If sentence is still too long, split it by CHUNK_SIZE
                            if len(sentence) > CHUNK_SIZE:
                                while sentence:
                                    chunks.append(sentence[:CHUNK_SIZE])
                                    sentence = sentence[CHUNK_SIZE:]
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    def _combine_audio_files(self, input_files, output_file):
        """Combine multiple audio files into one using ffmpeg"""
        if not input_files:
            raise ValueError("No input files provided for combining")
            
        if len(input_files) == 1:
            # If only one file, just copy it
            os.rename(input_files[0], output_file)
            return
            
        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            file_list_path = f.name
            for input_file in input_files:
                if os.path.exists(input_file):
                    f.write(f"file '{os.path.abspath(input_file)}'\n")
        
        try:
            # Use ffmpeg to concatenate files
            subprocess.run([
                'ffmpeg', '-f', 'concat', '-safe', '0', 
                '-i', file_list_path, '-c', 'copy', output_file
            ], check=True, stderr=subprocess.PIPE)
            
        except subprocess.CalledProcessError as e:
            # If the simple concatenation fails, try a more robust method
            try:
                # Create the command to join files with re-encoding
                cmd = ['ffmpeg']
                
                # Add each input file
                for input_file in input_files:
                    cmd.extend(['-i', input_file])
                
                # Add filter complex to concatenate audio streams
                filter_complex = ''
                for i in range(len(input_files)):
                    filter_complex += f'[{i}:0]'
                filter_complex += f'concat=n={len(input_files)}:v=0:a=1[out]'
                
                cmd.extend(['-filter_complex', filter_complex])
                cmd.extend(['-map', '[out]', output_file])
                
                # Run the command
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
                
            except subprocess.CalledProcessError:
                # If all fails, just keep the first file
                if os.path.exists(input_files[0]):
                    os.rename(input_files[0], output_file)
                    print(f"Warning: Audio concatenation failed. Only the first segment was saved.")
        
        finally:
            # Clean up the temporary file list
            if os.path.exists(file_list_path):
                os.unlink(file_list_path)
    
    def convert_fiction_to_speech(self, input_file, output_dir, voice_roles, style_instructions, model=DEFAULT_MODEL):
        """Convert fiction text to speech with segment-specific style instructions"""
        # Extract the text from the file
        text = self.get_text_from_file(input_file)
        
        # Process the text into voice-assigned segments
        segments = self.process_text(text, voice_roles)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Base filename for segments
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Generate and store audio segments
        segment_files = []
        for i, (segment_type, voice, content) in enumerate(segments):
            if not content.strip():
                continue
            
            # Get the instruction for this segment type
            instruction = style_instructions.get(segment_type, "")
                
            segment_file = os.path.join(output_dir, f"{base_name}_seg{i}_{voice}.mp3")
            segment_files.append(segment_file)
            
            self.convert_with_chunking(content, segment_file, voice, instruction, model)
            
        # Combine all segments into a final audio file
        final_output = os.path.join(output_dir, f"{base_name}_complete.mp3")
        self._combine_audio_files(segment_files, final_output)
        
        # Clean up segment files
        for file in segment_files:
            if os.path.exists(file):
                os.remove(file)
                
        return final_output

# GUI Implementation
class EnhancedTTSApp:
    def __init__(self, root):
        self.root = root
        self.converter = EnhancedTTSConverter()
        self.voice_roles = VOICE_ROLES.copy()
        self.style_instructions = {
            "narrator": "",
            "dialog": "",
            "computer": ""
        }
        self.custom_styles = {
            "narrator": "",
            "dialog": "",
            "computer": ""
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface with landscape orientation"""
        self.root.title("Fiction Text-to-Speech - Segment-Specific Styles")
        self.root.geometry("950x680")
        
        # Configure main layout - split into left and right sides
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left and right frames
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # === LEFT SIDE: Settings and Controls ===
        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        browse_btn = ttk.Button(
            file_frame, 
            text="Browse File", 
            command=self.browse_file, 
            width=15
        )
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Output directory selection
        output_frame = ttk.LabelFrame(left_frame, text="Output Directory", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        browse_output_btn = ttk.Button(
            output_frame, 
            text="Select Folder", 
            command=self.browse_output_dir, 
            width=15
        )
        browse_output_btn.pack(side=tk.LEFT, padx=5)
        
        self.output_dir_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Voice configuration
        voice_frame = ttk.LabelFrame(left_frame, text="Voice Configuration", padding="10")
        voice_frame.pack(fill=tk.X, pady=5)
        
        # Create voice selection grid
        # Narrator voice
        ttk.Label(voice_frame, text="Narrator Voice:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.narrator_var = tk.StringVar(value=self.voice_roles["narrator"])
        narrator_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.narrator_var,
            values=VOICES,
            state="readonly",
            width=12
        )
        narrator_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Dialog voice
        ttk.Label(voice_frame, text="Dialog Voice:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.dialog_var = tk.StringVar(value=self.voice_roles["dialog"])
        dialog_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.dialog_var,
            values=VOICES,
            state="readonly",
            width=12
        )
        dialog_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Computer voice
        ttk.Label(voice_frame, text="Computer Voice:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.computer_var = tk.StringVar(value=self.voice_roles["computer"])
        computer_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.computer_var,
            values=VOICES,
            state="readonly",
            width=12
        )
        computer_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Model selection
        ttk.Label(voice_frame, text="TTS Model:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        models = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
        model_menu = ttk.Combobox(
            voice_frame,
            textvariable=self.model_var,
            values=models,
            state="readonly",
            width=12
        )
        model_menu.grid(row=1, column=3, padx=5, pady=5)
        
        # Style settings - Separate tabs for each segment type
        style_frame = ttk.LabelFrame(left_frame, text="Segment-Specific Style Instructions", padding="10")
        style_frame.pack(fill=tk.X, pady=5)
        
        # Create notebook for segment types
        style_notebook = ttk.Notebook(style_frame)
        style_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a frame for each segment type
        narrator_frame = ttk.Frame(style_notebook, padding=5)
        dialog_frame = ttk.Frame(style_notebook, padding=5)
        computer_frame = ttk.Frame(style_notebook, padding=5)
        
        style_notebook.add(narrator_frame, text="Narrator")
        style_notebook.add(dialog_frame, text="Dialog")
        style_notebook.add(computer_frame, text="Computer")
        
        # Narrator style settings
        ttk.Label(narrator_frame, text="Style Preset:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.narrator_style_var = tk.StringVar(value=STYLE_PRESETS["narrator"][0])
        narrator_style_combo = ttk.Combobox(
            narrator_frame,
            textvariable=self.narrator_style_var,
            values=STYLE_PRESETS["narrator"],
            state="readonly",
            width=40
        )
        narrator_style_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        narrator_style_combo.bind("<<ComboboxSelected>>", lambda e: self.on_style_selected("narrator"))
        
        ttk.Label(narrator_frame, text="Custom Style:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.narrator_custom_var = tk.StringVar()
        self.narrator_custom_entry = ttk.Entry(narrator_frame, textvariable=self.narrator_custom_var, width=40)
        self.narrator_custom_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.narrator_custom_entry.config(state="disabled")
        
        # Dialog style settings
        ttk.Label(dialog_frame, text="Style Preset:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.dialog_style_var = tk.StringVar(value=STYLE_PRESETS["dialog"][0])
        dialog_style_combo = ttk.Combobox(
            dialog_frame,
            textvariable=self.dialog_style_var,
            values=STYLE_PRESETS["dialog"],
            state="readonly",
            width=40
        )
        dialog_style_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        dialog_style_combo.bind("<<ComboboxSelected>>", lambda e: self.on_style_selected("dialog"))
        
        ttk.Label(dialog_frame, text="Custom Style:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.dialog_custom_var = tk.StringVar()
        self.dialog_custom_entry = ttk.Entry(dialog_frame, textvariable=self.dialog_custom_var, width=40)
        self.dialog_custom_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.dialog_custom_entry.config(state="disabled")
        
        # Computer style settings
        ttk.Label(computer_frame, text="Style Preset:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.computer_style_var = tk.StringVar(value=STYLE_PRESETS["computer"][0])
        computer_style_combo = ttk.Combobox(
            computer_frame,
            textvariable=self.computer_style_var,
            values=STYLE_PRESETS["computer"],
            state="readonly",
            width=40
        )
        computer_style_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        computer_style_combo.bind("<<ComboboxSelected>>", lambda e: self.on_style_selected("computer"))
        
        ttk.Label(computer_frame, text="Custom Style:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.computer_custom_var = tk.StringVar()
        self.computer_custom_entry = ttk.Entry(computer_frame, textvariable=self.computer_custom_var, width=40)
        self.computer_custom_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.computer_custom_entry.config(state="disabled")
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        preview_btn = ttk.Button(
            button_frame, 
            text="Preview Segmentation", 
            command=self.preview_segmentation
        )
        preview_btn.pack(side=tk.LEFT, padx=5)
        
        convert_btn = ttk.Button(
            button_frame, 
            text="Convert to Speech", 
            command=self.convert_file,
            style="Accent.TButton"
        )
        convert_btn.pack(side=tk.RIGHT, padx=5)
        
        # Style for accent button
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="blue")
        
        # Progress bar
        progress_frame = ttk.LabelFrame(left_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            mode="indeterminate", 
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Progress status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # === RIGHT SIDE: Preview and Logs ===
        # Text preview
        preview_frame = ttk.LabelFrame(right_frame, text="Text Preview & Segmentation", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
        # Log text area
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
    
    def on_style_selected(self, segment_type):
        """Handle style preset selection for a specific segment type"""
        if segment_type == "narrator":
            selected_style = self.narrator_style_var.get()
            if selected_style == "Custom style...":
                self.narrator_custom_entry.config(state="normal")
                self.style_instructions["narrator"] = self.narrator_custom_var.get()
            else:
                self.narrator_custom_entry.config(state="disabled")
                if selected_style == "Default (no style instruction)":
                    self.style_instructions["narrator"] = ""
                else:
                    self.style_instructions["narrator"] = selected_style
        
        elif segment_type == "dialog":
            selected_style = self.dialog_style_var.get()
            if selected_style == "Custom style...":
                self.dialog_custom_entry.config(state="normal")
                self.style_instructions["dialog"] = self.dialog_custom_var.get()
            else:
                self.dialog_custom_entry.config(state="disabled")
                if selected_style == "Default (no style instruction)":
                    self.style_instructions["dialog"] = ""
                else:
                    self.style_instructions["dialog"] = selected_style
        
        elif segment_type == "computer":
            selected_style = self.computer_style_var.get()
            if selected_style == "Custom style...":
                self.computer_custom_entry.config(state="normal")
                self.style_instructions["computer"] = self.computer_custom_var.get()
            else:
                self.computer_custom_entry.config(state="disabled")
                if selected_style == "Default (no style instruction)":
                    self.style_instructions["computer"] = ""
                else:
                    self.style_instructions["computer"] = selected_style
    
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def browse_file(self):
        """Handle file browsing"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("Word files", "*.docx")]
        )
        
        if filepath:
            self.file_path_var.set(filepath)
            self.log_message(f"Selected file: {os.path.basename(filepath)}")
            
            # Default output directory to same as input file
            default_output = os.path.dirname(filepath)
            if not self.output_dir_var.get():
                self.output_dir_var.set(default_output)
    
    def browse_output_dir(self):
        """Handle output directory selection"""
        output_dir = filedialog.askdirectory()
        
        if output_dir:
            self.output_dir_var.set(output_dir)
            self.log_message(f"Selected output directory: {output_dir}")
    
    def update_style_instructions(self):
        """Update all style instructions from UI"""
        # Narrator style
        selected_style = self.narrator_style_var.get()
        if selected_style == "Custom style...":
            self.style_instructions["narrator"] = self.narrator_custom_var.get()
        elif selected_style == "Default (no style instruction)":
            self.style_instructions["narrator"] = ""
        else:
            self.style_instructions["narrator"] = selected_style
            
        # Dialog style
        selected_style = self.dialog_style_var.get()
        if selected_style == "Custom style...":
            self.style_instructions["dialog"] = self.dialog_custom_var.get()
        elif selected_style == "Default (no style instruction)":
            self.style_instructions["dialog"] = ""
        else:
            self.style_instructions["dialog"] = selected_style
            
        # Computer style
        selected_style = self.computer_style_var.get()
        if selected_style == "Custom style...":
            self.style_instructions["computer"] = self.computer_custom_var.get()
        elif selected_style == "Default (no style instruction)":
            self.style_instructions["computer"] = ""
        else:
            self.style_instructions["computer"] = selected_style
    
    def preview_segmentation(self):
        """Preview text segmentation"""
        filepath = self.file_path_var.get()
        
        if not filepath:
            messagebox.showwarning("Warning", "Please select a file first")
            return
            
        try:
            # Get text from file
            text = self.converter.get_text_from_file(filepath)
            
            # Update voice settings
            self.voice_roles["narrator"] = self.narrator_var.get()
            self.voice_roles["dialog"] = self.dialog_var.get()
            self.voice_roles["computer"] = self.computer_var.get()
            
            # Update style instructions
            self.update_style_instructions()
            
            # Process text to get segments
            segments = self.converter.process_text(text, self.voice_roles)
            
            # Display segmented text with voice and style assignments
            self.preview_text.delete(1.0, tk.END)
            
            for segment_type, voice, content in segments:
                # Get the style for this segment
                style = self.style_instructions.get(segment_type, "")
                style_info = f" - Style: {style}" if style else ""
                
                # Add segment header
                segment_header = f"[Type: {segment_type} | Voice: {voice}{style_info}]\n"
                self.preview_text.insert(tk.END, segment_header, "segment_header")
                
                # Add content
                self.preview_text.insert(tk.END, f"{content}\n\n")
            
            # Configure tag for segment headers
            self.preview_text.tag_configure("segment_header", foreground="blue", font=("TkDefaultFont", 9, "bold"))
            
            self.log_message(f"Identified {len(segments)} text segments")
            
        except Exception as e:
            self.log_message(f"Error previewing segmentation: {str(e)}")
            messagebox.showerror("Error", f"Error previewing segmentation: {str(e)}")
    
    def convert_file(self):
        """Convert the file to speech"""
        filepath = self.file_path_var.get()
        output_dir = self.output_dir_var.get()
        
        if not filepath:
            messagebox.showwarning("Warning", "Please select a file first")
            return
            
        if not output_dir:
            messagebox.showwarning("Warning", "Please select an output directory")
            return
        
        # Update voice settings
        self.voice_roles["narrator"] = self.narrator_var.get()
        self.voice_roles["dialog"] = self.dialog_var.get()
        self.voice_roles["computer"] = self.computer_var.get()
        
        # Update style instructions
        self.update_style_instructions()
        
        selected_model = self.model_var.get()
        
        # Start conversion in a separate thread
        threading.Thread(
            target=self._run_conversion, 
            args=(filepath, output_dir, selected_model), 
            daemon=True
        ).start()
    
    def _run_conversion(self, filepath, output_dir, model):
        """Run the conversion process"""
        try:
            self.progress_bar.start()
            self.log_message("Starting conversion process...")
            
            # Log the configuration
            self.log_message(f"Using narrator voice: {self.voice_roles['narrator']}")
            self.log_message(f"Using dialog voice: {self.voice_roles['dialog']}")
            self.log_message(f"Using computer voice: {self.voice_roles['computer']}")
            self.log_message(f"Using model: {model}")
            
            # Log style instructions
            for segment_type, instruction in self.style_instructions.items():
                if instruction:
                    self.log_message(f"Using {segment_type} style: '{instruction}'")
            
            # Convert the file
            final_output = self.converter.convert_fiction_to_speech(
                filepath, 
                output_dir, 
                self.voice_roles,
                self.style_instructions,
                model
            )
            
            self.progress_bar.stop()
            self.log_message(f"Conversion complete! Final audio saved to: {final_output}")
            messagebox.showinfo("Success", f"✅ Conversion complete!\nFinal audio saved to:\n{final_output}")
            
        except Exception as e:
            self.progress_bar.stop()
            error_msg = f"Error during conversion: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", f"❌ {error_msg}")

# Main application entry point
def main():
    root = tk.Tk()
    app = EnhancedTTSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()