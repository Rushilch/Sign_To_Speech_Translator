import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageSequence
import cv2
import numpy as np
import mediapipe as mp
import collections
import speech_recognition as sr
from googletrans import Translator
import threading
import re
import joblib
import time
import random
import queue
import asyncio
from gtts import gTTS
import pygame
import uuid
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()



# ================= CONFIG =================
CONF_THRESH = 0.6
STABLE_FRAMES = 10
PAUSE_FRAMES = 15
pygame.mixer.init()

MODEL_PATH = "hybrid_model.pkl"
LABELS_PATH = "hybrid_labels.txt"

ISL_ROOT = "images/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus"
SENTENCE_DIR = os.path.join(ISL_ROOT, "Sentence_Level")
WORD_DIR = os.path.join(ISL_ROOT, "Word_Level")
LETTER_DIR = os.path.join(ISL_ROOT, "Letter_Level")

SCRIPT_RANGES = {
    "en": [("a", "z")],  # Latin handled separately

    "hi": [("\u0900", "\u097F")],  # Devanagari
    "mr": [("\u0900", "\u097F")],  # Marathi uses Devanagari

    "bn": [("\u0980", "\u09FF")],  # Bengali
    "gu": [("\u0A80", "\u0AFF")],  # Gujarati
    "kn": [("\u0C80", "\u0CFF")],  # Kannada
    "te": [("\u0C00", "\u0C7F")],  # Telugu
    "ta": [("\u0B80", "\u0BFF")],  # Tamil
}

# Check if directories exist
for dir_path in [SENTENCE_DIR, WORD_DIR, LETTER_DIR]:
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

# Language options
LANGUAGES = {
    "English": {"code": "en", "speech": "en-IN"},
    "Hindi": {"code": "hi", "speech": "hi-IN"},
    "Telugu": {"code": "te", "speech": "te-IN"},
    "Tamil": {"code": "ta", "speech": "ta-IN"},
    "Bengali": {"code": "bn", "speech": "bn-IN"},
    "Marathi": {"code": "mr", "speech": "mr-IN"},
    "Gujarati": {"code": "gu", "speech": "gu-IN"},
    "Kannada": {"code": "kn", "speech": "kn-IN"},
}
# =========================================

def norm(txt):
    return re.sub(r"[^a-z ]", "", txt.lower()).strip()

def norm_key(txt):
    return txt.replace(" ", "_")

def normalize_hand(h):
    h = h.copy()
    h -= h.mean(axis=0)
    s = np.linalg.norm(h, axis=1).max()
    if s > 1e-6:
        h /= s
    return h

def resize_image_to_fit(image, max_w, max_h):
    """Helper to resize PIL image maintaining aspect ratio"""
    w, h = image.size
    ratio = min(max_w/w, max_h/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def speak_google_tts(text, lang_code):
    if not text.strip():
        return

    filename = f"tts_{uuid.uuid4().hex}.mp3"

    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print("Google TTS error:", e)

    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()  # 🔑 VERY IMPORTANT
            time.sleep(0.2)              # allow OS to release file
        except:
            pass

        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print("Cleanup error:", e)


def natural_sort_key(s):
    """Sort strings containing numbers naturally (1, 2, 10 instead of 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ================= SIGN DATABASE HELPER =================
class SignDatabase:
    """Helper class to manage sign language image database"""
    
    @staticmethod
    def get_display_sequence(text):
        """
        Returns a list of items to play.
        Item format: {'type': 'sentence'|'word'|'letter', 'path': str, 'text': str, 'frames': list}
        """
        sequence = []
        
        # 1. Check Sentence Level first
        sentence_key = norm_key(text)
        sentence_path = os.path.join(SENTENCE_DIR, sentence_key)
        
        if os.path.exists(sentence_path):
            images = []
            image_files = [f for f in os.listdir(sentence_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            image_files.sort(key=natural_sort_key)
            
            if image_files:
                images = [os.path.join(sentence_path, f) for f in image_files]
                # Return just the sentence sequence
                return [{'type': 'sentence', 'path': sentence_path, 'text': text, 'frames': images}]
        
        # 2. Word Level with Letter Fallback
        words = text.split()
        for word in words:
            word_key = word.lower()
            word_folder = os.path.join(WORD_DIR, word_key)
            
            found_word = False
            if os.path.exists(word_folder) and os.path.isdir(word_folder):
                image_files = [f for f in os.listdir(word_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                if image_files:
                    # Pick random image for the word
                    img_path = os.path.join(word_folder, random.choice(image_files))
                    sequence.append({'type': 'word', 'path': img_path, 'text': word, 'frames': None})
                    found_word = True
            
            # If word not found, spell it out (Letter Level)
            if not found_word:
                for letter in word:
                    letter_key = letter.lower()
                    # Check for various extensions
                    letter_path = None
                    for ext in ['.gif', '.jpg', '.png', '.jpeg']:
                        p = os.path.join(LETTER_DIR, f"{letter_key}{ext}")
                        if os.path.exists(p):
                            letter_path = p
                            break
                    
                    if letter_path:
                        sequence.append({'type': 'letter', 'path': letter_path, 'text': letter, 'frames': None})
        
        return sequence

# ================= SIGN PLAYER =================
class SignPlayer:
    def __init__(self, label, status_callback=None):
        self.label = label
        self.running = False
        self.frames = []
        self.idx = 0
        self.gif_frames = []
        self.gif_idx = 0
        self.gif_running = False
        self.status_callback = status_callback
        
    def update_status(self, message, color=None):
        if self.status_callback:
            self.status_callback(message, color)
            
    def stop(self):
        self.running = False
        self.gif_running = False
        
    def show_static(self, image_path):
        """Display a static image (used for words)"""
        self.stop()
        try:
            img = Image.open(image_path)
            img = resize_image_to_fit(img, 400, 350)
            tkimg = ImageTk.PhotoImage(img)
            self.label.config(image=tkimg)
            self.label.image = tkimg
        except Exception as e:
            print(f"Error showing static image: {e}")

    def play_frames(self, frames, delay=100):
        """Play a list of image paths (used for sentences)"""
        self.stop()
        self.frames = frames
        self.idx = 0
        self.running = True
        self._next_frame_seq(delay)
        
    def _next_frame_seq(self, delay):
        if not self.running or self.idx >= len(self.frames):
            return
        try:
            img = Image.open(self.frames[self.idx])
            img = resize_image_to_fit(img, 400, 350)
            tkimg = ImageTk.PhotoImage(img)
            self.label.config(image=tkimg)
            self.label.image = tkimg
            self.idx += 1
            self.label.after(delay, lambda: self._next_frame_seq(delay))
        except Exception as e:
            print(f"Error frame seq: {e}")
            self.running = False

    def play_gif(self, gif_path):
        """Play a GIF file (used for letters)"""
        self.stop()
        try:
            gif = Image.open(gif_path)
            self.gif_frames = []
            for frame in ImageSequence.Iterator(gif):
                resized = resize_image_to_fit(frame.convert("RGB"), 400, 350)
                self.gif_frames.append(ImageTk.PhotoImage(resized))
            
            if not self.gif_frames:
                # Fallback if single frame loaded
                self.show_static(gif_path)
                return

            self.gif_idx = 0
            self.gif_running = True
            self._next_gif_frame()
        except Exception as e:
            print(f"Error GIF: {e}")

    def _next_gif_frame(self):
        if not self.gif_running: return
        try:
            if self.gif_idx >= len(self.gif_frames): self.gif_idx = 0
            if self.gif_frames:
                self.label.config(image=self.gif_frames[self.gif_idx])
                self.label.image = self.gif_frames[self.gif_idx]
                self.gif_idx = (self.gif_idx + 1) % len(self.gif_frames)
                self.label.after(100, self._next_gif_frame)
        except: self.gif_running = False

# ================= MODERN UI COMPONENTS =================
class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        self.bg_color = kwargs.pop('bg_color', '#3498DB')
        self.fg_color = kwargs.pop('fg_color', '#FFFFFF')
        self.hover_color = kwargs.pop('hover_color', '#2980B9')
        self.font = kwargs.pop('font', ("Segoe UI", 11))
        self.padding = kwargs.pop('padding', (12, 6))
        
        super().__init__(master, **kwargs)
        self.config(bg=self.bg_color, fg=self.fg_color, font=self.font, bd=0, relief="flat", cursor="hand2", padx=self.padding[0], pady=self.padding[1])
        self.bind("<Enter>", lambda e: self.config(bg=self.hover_color))
        self.bind("<Leave>", lambda e: self.config(bg=self.bg_color))

class Card(tk.Frame):
    def __init__(self, parent, title=None, bg="#020617"):
        super().__init__(
            parent,
            bg=bg,
            bd=0,
            highlightthickness=1,
            highlightbackground="#1E293B"
        )

        self.inner = tk.Frame(self, bg=bg)
        self.inner.pack(fill="both", expand=True, padx=18, pady=14)

        if title:
            tk.Label(
                self.inner,
                text=title,
                font=("Segoe UI", 10, "bold"),
                fg="#94A3B8",
                bg=bg
            ).pack(anchor="w", pady=(0, 10))



# ================= MAIN APP =================
class VaaniVerse(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VaaniVerse – Voice ↔ Sign ISL Interpreter")
        self.geometry("1280x800")
        self.minsize(1000, 700) 
        self.COLORS = {
        "primary": "#0F172A",     # header
        "secondary": "#2563EB",   # blue buttons
        "accent": "#22C55E",      # green highlights
        "background": "#020617",  # main bg
        "surface": "#020617",     # cards bg
        "card": "#020617",
        "text": "#E5E7EB",
        "muted": "#94A3B8",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "dark": "#020617",
        "gray": "#64748B",
        "info": "#2563EB"
        }
        self.configure(bg=self.COLORS['background'])
        
        # Init Helpers
        self.sign_db = SignDatabase()
        
        # Check Dependencies
        missing = []
        if not os.path.exists(MODEL_PATH): missing.append(f"Model: {MODEL_PATH}")
        if not os.path.exists(LABELS_PATH): missing.append(f"Labels: {LABELS_PATH}")
        if missing: messagebox.showwarning("Missing Files", "\n".join(missing))

        # AI & Speech
        try:
            self.model = joblib.load(MODEL_PATH)
            self.labels = open(LABELS_PATH).read().splitlines()
        except: self.model = None; self.labels = []
        
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1
        )

        self.draw = mp.solutions.drawing_utils
        self.recognizer = sr.Recognizer()
        
        # Check TTS availability
        self.tts_available = False
        try:
            import pyttsx3
            # Check if engine can be init
            engine = pyttsx3.init()
            self.tts_available = True
        except: 
            self.tts_available = False
        
        self.translator = Translator()
        self.message_queue = queue.Queue()
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # App State
        self.running = False
        self.is_listening = False
        self.prev_lh = None
        self.prev_rh = None
        self.pred_history = collections.deque(maxlen=STABLE_FRAMES)
        self.pause_counter = 0
        self.word_locked = False
        self.current_word = ""
        self.sentence = []
        self.phonetic_sentence = ""
        self.translated_sentence = ""
        self.use_ai_sentence = tk.BooleanVar(value=False)
        
        # Build UI
        self.build_ui()
        self.player = SignPlayer(self.sign_img, self.update_speak_status)
        self.check_queue()
        style = ttk.Style(self)
        style.theme_use("default")

        style.configure(
            "TNotebook",
            background=self.COLORS["background"],
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 11),
            padding=[16, 8],
            background="#020617",
            foreground="#94A3B8"
        )

        style.map(
            "TNotebook.Tab",
            background=[("selected", "#020617")],
            foreground=[("selected", "#F8FAFC")]
        )

        style.configure(
            "TCombobox",
            padding=6,
            fieldbackground="#020617",
            background="#020617",
            foreground="#F8FAFC"
        )

    def _draw_ai_toggle(self):
        c = self.ai_toggle_btn
        c.delete("all")

        on = self.use_ai_sentence.get()

        # background
        c.create_rounded_rect = lambda x1,y1,x2,y2,r,**kw: c.create_polygon(
            x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
            x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
            x1,y2, x1,y2-r, x1,y1+r, x1,y1,
            smooth=True, **kw
        )

        bg = "#22C55E" if on else "#374151"
        c.create_rounded_rect(2,2,44,22,10, fill=bg, outline="")

        knob_x = 26 if on else 4

        c.create_oval(knob_x, 4, knob_x+16, 20, fill="#F8FAFC", outline="")

    def _toggle_ai(self, event=None):
        self.use_ai_sentence.set(not self.use_ai_sentence.get())
        self._draw_ai_toggle()
        self.after(10, self._draw_ai_toggle)

    def check_queue(self):
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg[0] == 'status': self.update_speak_status(msg[1], msg[2])
                elif msg[0] == 'text': self.process_speech_text(msg[1])
        except queue.Empty: pass
        finally: self.after(100, self.check_queue)

    def build_ui(self):
        
        # ---- UI STATE VARIABLES (MUST EXIST BEFORE WIDGETS) ----
        self.speak_lang = tk.StringVar(value="Hindi")
        self.trans_lang = tk.StringVar(value="English")
        # Header
        h = tk.Frame(self, bg=self.COLORS['primary'], height=70)
        h.pack(fill="x", side="top"); h.pack_propagate(False)
        tk.Label(h, text="VaaniVerse", font=("Segoe UI", 24, "bold"), fg="#F8FAFC", bg=self.COLORS['primary']).pack(side="left", padx=20)
        
        # Tabs
        main = tk.Frame(self, bg=self.COLORS['background'])
        main.pack(fill="both", expand=True, padx=10, pady=10)
        tabs = ttk.Notebook(main)
        tabs.pack(fill="both", expand=True)
        
        # --- TAB 1: Speak -> Sign ---
        t1 = tk.Frame(tabs, bg=self.COLORS['background'])
        tabs.add(t1, text="🎤 Speak to Sign")
        
        content = tk.Frame(t1, bg=self.COLORS['background'])
        content.pack(expand=True, fill="both", padx=30, pady=20)

        columns = tk.Frame(content, bg=self.COLORS['background'])
        columns.pack(fill="both", expand=True)

        left_col = Card(columns, "Speech Input")
        left_col.pack(side="left", fill="y", padx=15)

        right_col = Card(columns, "ISL Output")
        right_col.pack(side="right", fill="both", expand=True, padx=15)

        
        # Controls
        f = tk.Frame(left_col.inner, bg=self.COLORS['card'])
        f.pack(fill="x", pady=5)

        tk.Label(f, text="Language", bg=self.COLORS['card']).pack(anchor="w")
        ttk.Combobox(
            f,
            textvariable=self.speak_lang,
            values=list(LANGUAGES.keys()),
            width=18
        ).pack(pady=4)
        self.speak_status = tk.Label(
        left_col.inner,
        text="Ready",
        font=("Segoe UI", 12),
        fg=self.COLORS['success'],
        bg=self.COLORS['card']
        )
        self.speak_status.pack(pady=10)
        self.rec_text = tk.Label(
            left_col.inner,
            text="",
            font=("Segoe UI", 11, "italic"),
            fg=self.COLORS['muted'],
            bg=self.COLORS['card'],
            wraplength=300,
            justify="center"
        )
        self.rec_text.pack(pady=6)
        
        # Image
        img_f = tk.Frame(
        right_col.inner,
        bg="#1E272E",
        height=360
            )
        img_f.pack(fill="both", expand=True, pady=10)
        img_f.pack_propagate(False)

        self.sign_img = tk.Label(img_f, bg="#1E272E")
        self.sign_img.place(relx=0.5, rely=0.5, anchor="center")

        self.sign_label = tk.Label(
            right_col.inner,
            text="",
            font=("Segoe UI", 14, "bold"),
            fg=self.COLORS['accent'],
            bg=self.COLORS['card']
        )
        self.sign_label.pack(pady=8)

    
        # Buttons (CREATE FIRST, THEN PACK)
        self.speak_btn = ModernButton(
            left_col.inner,
            text="🎤 Start Speaking",
            command=self.start_listen,
            bg_color=self.COLORS['secondary']
        )
        self.speak_btn.pack(pady=8, fill="x")

        self.stop_btn = ModernButton(
            left_col.inner,
            text="⏹ Stop",
            command=self.stop_listen,
            bg_color=self.COLORS['danger']
        )
        # stop button hidden initially (same behavior as before)


        
        # --- TAB 2: Sign -> Speech ---
        t2 = tk.Frame(tabs, bg=self.COLORS['background'])
        tabs.add(t2, text="✋ Sign to Speech")
        
        split = tk.Frame(t2, bg=self.COLORS['background'])
        split.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Camera
        left = Card(split, "Live Camera")
        left.pack(side="left", fill="both", expand=True, padx=10)

        
        lb = tk.Frame(left, bg=self.COLORS['background']); lb.pack(side="bottom", fill="x", pady=10)
        ModernButton(lb, text="▶ Start", command=self.start_cam, bg_color=self.COLORS['success']).pack(side="left", fill="x", expand=True, padx=2)
        ModernButton(lb, text="⏹ Stop", command=self.stop_cam, bg_color=self.COLORS['danger']).pack(side="left", fill="x", expand=True, padx=2)
        
        ls = tk.LabelFrame(
            lb,
            text="Settings",
            bg=self.COLORS['background'],
            fg=self.COLORS['muted'],      # 👈 title text color
            labelanchor="n"
        )
        ls.pack(fill="x", pady=5)
 
        tk.Label(ls, text="Translate To:", bg=self.COLORS['background']).pack(side="left", padx=5)
        ttk.Combobox(ls, textvariable=self.trans_lang, values=list(LANGUAGES.keys()), width=15).pack(side="right", padx=5)
        
        self.vid_con = tk.Frame(
            left.inner,
            bg="#1E272E",
            width=640,
            height=420
        )
        self.vid_con.pack(pady=10)
        self.vid_con.pack_propagate(False)  # 🔑 PREVENT RESIZE


        self.vid = tk.Label(self.vid_con, bg="#1E272E")
        self.vid.pack(fill="both", expand=False)

        
        # Right: Output
        right = Card(split, "Recognized Output")
        right.pack(side="right", fill="y", padx=10)

        # ⭐ Modern AI toggle (switch style)

        toggle_row = tk.Frame(right.inner, bg=self.COLORS["card"])
        toggle_row.pack(fill="x", pady=(4, 8))

        tk.Label(
            toggle_row,
            text="AI Sentence Enhancement",
            font=("Segoe UI", 10, "bold"),
            fg=self.COLORS["muted"],
            bg=self.COLORS["card"]
        ).pack(side="left")


        self.ai_toggle_btn = tk.Canvas(
            toggle_row,
            width=46,
            height=24,
            bg=self.COLORS["card"],
            highlightthickness=0,
            cursor="hand2"
        )
        self.ai_toggle_btn.pack(side="right")

        self._draw_ai_toggle()

        self.ai_toggle_btn.bind("<Button-1>", self._toggle_ai)

        
        rb = tk.Frame(right, bg=self.COLORS['background']); rb.pack(side="bottom", fill="x", pady=10)
        ModernButton(rb, text="🔊 Speak (English)", command=lambda: self.speak_txt("eng"), bg_color=self.COLORS['accent']).pack(fill="x", pady=2)
        ModernButton(rb, text="🗣️ Speak(Translated))", command=lambda: self.speak_txt("phonetic"), bg_color=self.COLORS['secondary']).pack(fill="x", pady=2)
        ModernButton(rb, text="🌐 Translate", command=self.perform_translation, bg_color=self.COLORS['warning']).pack(fill="x", pady=2)
        ModernButton(rb, text="🧹 Clear", command=self.clear_text, bg_color=self.COLORS['danger']).pack(fill="x", pady=2)
        
        out = tk.LabelFrame(right, text="Output", bg=self.COLORS['background']); out.pack(side="top", fill="both", expand=True)
        
        self.w_lbl = self.make_card(out, "Current Sign:", self.COLORS['secondary'], size=16)

        # ⭐ RAW sentence (model output)
        self.s_lbl = self.make_card(out, "Raw Sentence:", self.COLORS['text'], size=12)

        # ⭐ AI enhanced sentence
        self.ai_lbl = self.make_card(out, "AI Enhanced Sentence:", self.COLORS['success'], size=12)

        # Translation stays
        self.t_lbl = self.make_card(out, "Translation:", self.COLORS['accent'], size=12)
        

    def make_card(self, parent, title, color, size=12):
        f = tk.Frame(parent, bg=self.COLORS['card'], bd=1, highlightbackground="#1E293B", highlightthickness=1)
        f.pack(fill="x", padx=10, pady=5)
        tk.Label(f, text=title, font=("Segoe UI", 9), bg=self.COLORS['card'], fg=self.COLORS['muted']).pack(anchor="w", padx=5)
        l = tk.Label(f, text="...", font=("Segoe UI", size, "bold"), bg=self.COLORS['card'], fg=color, wraplength=280, justify="left")
        l.pack(fill="x", padx=5, pady=5)
        return l

    # ================= LOGIC =================
    def update_speak_status(self, msg, color='info'):
        c = self.COLORS.get(color, self.COLORS['text'])
        self.speak_status.config(text=msg, fg=c)

    def start_listen(self):
        self.is_listening = True
        self.speak_btn.pack_forget(); self.stop_btn.pack(pady=10, ipadx=20)
        self.update_speak_status("Listening...", 'info')
        threading.Thread(target=self.listen_thread, daemon=True).start()

    def stop_listen(self):
        self.is_listening = False
        self.stop_btn.pack_forget(); self.speak_btn.pack(pady=10, ipadx=20)
        self.update_speak_status("Ready", 'success')

    def listen_thread(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)

                # Try selected language first
                selected_lang = LANGUAGES[self.speak_lang.get()]["speech"]

                try:
                    text = self.recognizer.recognize_google(audio, language=selected_lang)
                except:
                    # fallback to English auto
                    text = self.recognizer.recognize_google(audio)

                self.message_queue.put(('text', text))

        except Exception as e:
            self.message_queue.put(('status', f"Error: {e}", 'danger'))

        finally:
            self.after(0, self.stop_listen)


    def translate_wrapper(self, text, src_code=None, dest_code='en'):
            if not text:
                return None
            try:
                translator = Translator()
                result = translator.translate(text, src=src_code, dest=dest_code)

                # googletrans sometimes returns coroutine
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)

                # Safety check
                if not hasattr(result, "text") or not result.text:
                    return None

                return result
            except Exception as e:
                print(f"[TRANSLATE ERROR] {e}")
                return None

    def groq_refine_sentence(self, text):
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Convert sign language word sequence into natural spoken English.\n"
                            "Fix grammar and meaning.\n"
                            "Return ONE short sentence only."
                        )
                    },
                    {"role": "user", "content": text}
                ],
            )

            refined = resp.choices[0].message.content.strip()
            print("AI refined:", refined)
            return refined

        except Exception as e:
            print("Groq refine error:", e)
            return text

    def groq_language_ratio(self, text, selected_lang_code):
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Detect dominant language of the sentence and return JSON only.\n"
                            "Fields: dominant_language (ISO code), ratio (0-1).\n"
                            "Example: {\"dominant_language\":\"en\",\"ratio\":0.8}"
                        )
                    },
                    {"role": "user", "content": text}
                ],
            )

            raw = resp.choices[0].message.content.strip()

            data = json.loads(raw)

            dominant = data.get("dominant_language")
            ratio = float(data.get("ratio", 0))

            print("Groq language:", dominant, ratio)

            if dominant != selected_lang_code or ratio < 0.5:
                return False

            return True

        except Exception as e:
            print("Groq language error:", e)
            return True  # fail open so UX not blocked


    def process_speech_text(self, text):
        self.rec_text.config(text=f"Said: {text}")

        self.update_speak_status("Translating to English...", 'info')

        src_lang = self.speak_lang.get()
        src_code = LANGUAGES[src_lang]['code']

        # ⭐ Strict language dominance check (50%)
        if not self.groq_language_ratio(text, src_code):
            self.update_speak_status("Language mismatch — please speak selected language", "danger")
            return


        # 🔁 STEP 1: Translate spoken text → English
        res = self.translate_wrapper(text, src_code=src_code, dest_code='en')

        if res and res.text:
            english_text = res.text
        else:
            # Fallback: assume already English
            english_text = text

        # 🔁 STEP 2: Normalize English ONLY
        english_text = re.sub(r"[^a-zA-Z ]", "", english_text).lower().strip()

        if not english_text:
            self.update_speak_status("Translation failed", 'error')
            return

        self.update_speak_status(f"Playing signs for: {english_text}", 'success')

        # 🔁 STEP 3: Play signs using ENGLISH ONLY
        self.play_sign_sequence(english_text)


    def play_sign_sequence(self, text):
        norm_text = re.sub(r"[^a-z ]", "", text.lower()).strip()
        playlist = self.sign_db.get_display_sequence(norm_text)
        
        if not playlist:
            self.update_speak_status("No signs found", 'warning')
            return
            
        self.update_speak_status(f"Playing signs for: {text}", 'success')
        
        def play_next_item(index=0):
            if index >= len(playlist):
                self.update_speak_status("Playback Complete", 'success')
                self.sign_label.config(text="")
                self.player.stop()
                return

            item = playlist[index]
            self.sign_label.config(text=f"{item['type'].title()}: {item['text']}")
            
            wait_time = 2000 
            
            if item['type'] == 'sentence':
                self.player.play_frames(item['frames'])
                wait_time = len(item['frames']) * 100 + 1000 
            elif item['type'] == 'word':
                self.player.show_static(item['path'])
                wait_time = 2000
            elif item['type'] == 'letter':
                self.player.play_gif(item['path'])
                wait_time = 1500 
            
            self.after(wait_time, lambda: play_next_item(index + 1))

        play_next_item(0)

    # --- CAM LOGIC ---
    def start_cam(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.cam_loop, daemon=True).start()

    def stop_cam(self): self.running = False

    def cam_loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, fr = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            hres = self.hands.process(rgb); fres = self.face.process(rgb)
            det = False
            
            if fres.multi_face_landmarks and hres.multi_hand_landmarks:
                pts = []
                for lm in hres.multi_hand_landmarks:
                    pts.append(np.array([[x.x, x.y] for x in lm.landmark]))
                    self.draw.draw_landmarks(fr, lm, mp.solutions.hands.HAND_CONNECTIONS)
                
                if len(pts) == 1: pts *= 2
                pts.sort(key=lambda p: p[:,0].mean())
                nose = fres.multi_face_landmarks[0].landmark[1]
                
                l, r = normalize_hand(pts[0]), normalize_hand(pts[1])
                lc, rc = l.mean(0), r.mean(0)
                fc = np.array([nose.x, nose.y])
                
                feat = np.concatenate([l.flatten(), r.flatten(), [np.linalg.norm(lc-fc)], [np.linalg.norm(rc-fc)], np.zeros(8)])[None,:]
                
                if self.model:
                    pred = self.model.predict(feat)[0]
                    self.pred_history.append(pred)
                    if self.pred_history.count(pred) >= STABLE_FRAMES and not self.word_locked:
                        self.current_word = self.labels[pred]
                        self.w_lbl.after(0, lambda w=self.current_word: self.w_lbl.config(text=w, fg=self.COLORS['success']))
                        self.word_locked = True; det = True
            
            if det: self.pause_counter = 0
            else: self.pause_counter += 1
            
            if self.pause_counter >= PAUSE_FRAMES and self.current_word:
                self.sentence.append(self.current_word)
                self.s_lbl.after(0, lambda s=" ".join(self.sentence): self.s_lbl.config(text=s))
                self.after(0, self.perform_translation)
                self.current_word = ""; self.word_locked = False; self.pred_history.clear(); self.pause_counter = 0
            
            # Display
            w, h = self.vid_con.winfo_width(), self.vid_con.winfo_height()
            if w > 10 and h > 10:
                ih, iw, _ = fr.shape
                s = min(w/iw, h/ih)
                fr = cv2.resize(fr, (int(iw*s), int(ih*s)))
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            self.vid.after(0, lambda i=img: self.vid.config(image=i)); self.vid.image = img
        cap.release()

    def perform_translation(self):
        """Translates recognized English sentence to target language and gets pronunciation"""
        if not self.sentence: return
        raw_txt = " ".join(self.sentence)

        # show raw always
        self.s_lbl.config(text=raw_txt)

        # AI enhancement
        if self.use_ai_sentence.get():
            enhanced_txt = self.groq_refine_sentence(raw_txt)
        else:
            enhanced_txt = raw_txt

        # show enhanced
        self.ai_lbl.config(text=enhanced_txt)

        txt = enhanced_txt

        dest_code = LANGUAGES[self.trans_lang.get()]['code']
        
        try:
            # Use safe wrapper to translate English -> Target
            res = self.translate_wrapper(txt, src_code='en', dest_code=dest_code)
            
            if res:
                self.translated_sentence = res.text
                
                self.phonetic_sentence = res.pronunciation if res.pronunciation else res.text
                
                self.t_lbl.config(text=self.translated_sentence, fg=self.COLORS['accent'])
                if hasattr(self, "p_lbl"):
                    self.p_lbl.config(text=self.phonetic_sentence, fg=self.COLORS['success'])

            else:
                self.t_lbl.config(text="Error", fg=self.COLORS['danger'])
                
        except Exception as e: 
            print(f"Translation Logic Error: {e}")

    def speak_txt(self, mode):
        if not self.sentence:
            return

        if mode == "phonetic":
            lang_code = LANGUAGES[self.trans_lang.get()]["code"]
            text = self.translated_sentence
        else:
            lang_code = "en"
            text = " ".join(self.sentence)

        threading.Thread(
            target=speak_google_tts,
            args=(text, lang_code),
            daemon=True
        ).start()



    def _run_tts(self, text):
        try:
            # Initialize engine inside thread to prevent blocking/freezing
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def clear_text(self):
        self.sentence = []; self.phonetic_sentence = ""; self.translated_sentence = ""
        self.w_lbl.config(text="..."); self.s_lbl.config(text="...")
        self.t_lbl.config(text="...")

        if hasattr(self, "p_lbl"):
            self.p_lbl.config(text="...")


if __name__ == "__main__":
    app = VaaniVerse()
    app.mainloop()