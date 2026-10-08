import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import re
import threading
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from scipy.stats import skew, kurtosis
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- 1. CONFIGURATION & SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. BACKEND LOGIC ---
model = None
tfidf_vectorizer = None

# Model Yükleme
try:
    model = joblib.load('fake_news_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ System ready. Models loaded.")
except Exception as e:
    print(f"⚠️ WARNING: Model files not found.\nError: {e}")

# --- URL FETCH FUNCTION ---
def fetch_news_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Hata: Sayfa yüklenemedi (Kod: {response.status_code})"

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').get_text().strip() if soup.find('h1') else "Başlık Bulunamadı"
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text()) > 20])
        
        if not text:
            return None, "İçerik çekilemedi veya metin çok kısa."
            
        return title, text

    except Exception as e:
        return None, str(e)

# --- HELPER FUNCTIONS (EKSİKLER EKLENDİ) ---
def caps_ratio(text): 
    text = str(text)
    return sum(1 for c in text if c.isupper()) / max(len(text), 1)

def count_exclamations(text):
    return str(text).count('!')

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

def get_statistical_moments(text):
    if not isinstance(text, str): return [0, 0]
    word_lengths = [len(w) for w in text.split()]
    if len(word_lengths) < 2: return [0, 0]
    return [skew(word_lengths), kurtosis(word_lengths)]

def get_lexical_features(text):
    if not isinstance(text, str) or len(text) == 0: return [0, 0]
    words = text.split()
    if len(words) == 0: return [0, 0]
    return [len(set(words))/len(words), sum(len(w) for w in words)/len(words)]

def clean_text_final(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def generate_forensic_insights(title, text, stats_features):
    insights = []
    # stats_features artık 10 elemanlı, doğru indexleri kullanalım
    # 0: title_caps, 1: title_excl, 2: title_sent, 3: text_caps...
    t_caps = stats_features[0, 0] 
    
    subjectivity = TextBlob(text).sentiment.subjectivity
    text_lower = text.lower()
    title_lower = title.lower()
    
    fake_triggers = ['video', 'watch', 'wow', 'breaking', 'exposed', 'shocking', 'share', 'viral', 'secret', 'plot']
    found_fakes = list(set([w for w in fake_triggers if w in text_lower or w in title_lower]))

    if found_fakes:
        insights.append(f"🚩 TRIGGERS: Found {len(found_fakes)} suspicious term(s): {', '.join(found_fakes[:3])}")
    
    excl_count = text.count('!') + title.count('!')
    if excl_count > 2:
        insights.append(f"🔥 EMOTION: High usage of exclamation marks ({excl_count} found).")

    if t_caps > 0.20:
        insights.append(f"📢 HEADLINE: Aggressive capitalization ({t_caps*100:.1f}%).")

    if subjectivity > 0.6:
        insights.append(f"🎭 TONE: Highly subjective content ({subjectivity:.2f}/1.0).")
    
    if len(insights) < 1:
        insights.append("✅ GENERAL: No linguistic anomalies found.")

    return insights

# --- 3. MODERN GUI CLASS ---
class FakeNewsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CEN AI | Intelligent News Verification V2.1")
        self.geometry("1200x850")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar_frame = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="CEN AI 2.1", font=ctk.CTkFont(size=30, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.history_label = ctk.CTkLabel(self.sidebar_frame, text="RECENT SCANS", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray")
        self.history_label.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.history_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, width=220, height=400, fg_color="transparent")
        self.history_scroll.grid(row=2, column=0, padx=10)
        self.history_items = []

        self.theme_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light"], command=self.change_theme)
        self.theme_menu.grid(row=5, column=0, padx=20, pady=(50, 20), sticky="s")

        # --- MAIN AREA ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)

        # URL
        self.url_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.url_frame.pack(fill="x", pady=(0, 20))
        
        self.url_entry = ctk.CTkEntry(self.url_frame, placeholder_text="Paste Article URL here to auto-fetch...", height=40)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.fetch_btn = ctk.CTkButton(self.url_frame, text="⬇ FETCH URL", width=120, height=40, command=self.start_fetch_thread, fg_color="#E0aaff", text_color="black")
        self.fetch_btn.pack(side="right")

        # INPUTS
        self.title_label = ctk.CTkLabel(self.main_frame, text="HEADLINE", font=ctk.CTkFont(size=12, weight="bold"))
        self.title_label.pack(anchor="w")
        self.title_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Title...", height=40)
        self.title_entry.pack(fill="x", pady=(0, 10))

        self.text_label = ctk.CTkLabel(self.main_frame, text="CONTENT", font=ctk.CTkFont(size=12, weight="bold"))
        self.text_label.pack(anchor="w")
        self.text_entry = ctk.CTkTextbox(self.main_frame, height=180)
        self.text_entry.pack(fill="x", pady=(0, 20))

        # BUTTONS
        self.btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", pady=(0, 20))
        
        self.analyze_btn = ctk.CTkButton(self.btn_frame, text="🔍 ANALYZE NEWS", height=50, font=ctk.CTkFont(size=16, weight="bold"), command=self.start_analysis_thread)
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.clear_btn = ctk.CTkButton(self.btn_frame, text="RESET", height=50, width=100, fg_color="transparent", border_width=2, command=self.clear_inputs)
        self.clear_btn.pack(side="right")

        # RESULTS
        self.result_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.result_frame.pack(fill="both", expand=True)

        self.verdict_box = ctk.CTkFrame(self.result_frame, corner_radius=15, fg_color=("gray85", "gray17"))
        self.verdict_box.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.verdict_icon = ctk.CTkLabel(self.verdict_box, text="⚖️", font=ctk.CTkFont(size=60))
        self.verdict_icon.pack(pady=(50, 20))
        
        self.verdict_label = ctk.CTkLabel(self.verdict_box, text="WAITING FOR INPUT", font=ctk.CTkFont(size=22, weight="bold"))
        self.verdict_label.pack()
        
        self.score_label = ctk.CTkLabel(self.verdict_box, text="Confidence: --%", font=ctk.CTkFont(size=16))
        self.score_label.pack(pady=(15, 0))

        self.insight_box = ctk.CTkFrame(self.result_frame, corner_radius=15)
        self.insight_box.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.insight_title = ctk.CTkLabel(self.insight_box, text="FORENSIC REPORT", font=ctk.CTkFont(size=16, weight="bold"))
        self.insight_title.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.insight_text = ctk.CTkTextbox(self.insight_box, fg_color="transparent", font=ctk.CTkFont(size=14))
        self.insight_text.pack(fill="both", expand=True, padx=15, pady=15)
        self.insight_text.insert("0.0", "AI Insights will appear here...")

    def change_theme(self, mode):
        ctk.set_appearance_mode(mode)

    def clear_inputs(self):
        self.title_entry.delete(0, "end")
        self.text_entry.delete("0.0", "end")
        self.url_entry.delete(0, "end")
        self.verdict_box.configure(fg_color=("gray85", "gray17"))
        self.verdict_label.configure(text="WAITING FOR INPUT", text_color="gray")
        self.verdict_icon.configure(text="⚖️")
        self.score_label.configure(text="Confidence: --%")
        self.insight_text.delete("0.0", "end")

    def add_to_history(self, title, is_fake):
        color = "#ff5555" if is_fake else "#55ff99"
        icon = "❌" if is_fake else "✅"
        short_title = (title[:25] + '...') if len(title) > 25 else title
        item = ctk.CTkButton(self.history_scroll, text=f"{icon} {short_title}", 
                             fg_color="transparent", border_width=1, border_color="gray",
                             anchor="w", height=30, font=ctk.CTkFont(size=11))
        item.pack(fill="x", pady=2)
        self.history_items.append(item)

    def start_fetch_thread(self):
        url = self.url_entry.get().strip()
        if not url: return
        self.fetch_btn.configure(state="disabled", text="FETCHING...")
        threading.Thread(target=self.fetch_url, args=(url,), daemon=True).start()

    def fetch_url(self, url):
        title, text = fetch_news_from_url(url)
        self.fetch_btn.configure(state="normal", text="⬇ FETCH URL")
        if title is None:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to fetch URL.\n{text}"))
        else:
            self.after(0, lambda: self.update_fields(title, text))

    def update_fields(self, title, text):
        self.title_entry.delete(0, "end")
        self.title_entry.insert(0, title)
        self.text_entry.delete("0.0", "end")
        self.text_entry.insert("0.0", text)

    def start_analysis_thread(self):
        self.analyze_btn.configure(state="disabled", text="ANALYZING...")
        threading.Thread(target=self.analyze_news, daemon=True).start()

    # --- KRİTİK DÜZELTME BURADA ---
    def analyze_news(self):
        title = self.title_entry.get().strip()
        text = self.text_entry.get("0.0", "end").strip()

        if not text:
            self.after(0, lambda: messagebox.showwarning("Input Error", "Please enter text or fetch a URL."))
            self.reset_btn()
            return

        if model is None:
            self.after(0, lambda: messagebox.showerror("System Error", "AI Model not loaded."))
            self.reset_btn()
            return

        try:
            # --- 1. TÜM 10 ÖZELLİĞİ HESAPLA (Modelin beklediği sıra ile) ---
            
            # Title Features
            t_caps = caps_ratio(title)
            t_excl = count_exclamations(title)
            t_sent = get_sentiment(title)
            
            # Text Features
            txt_caps = caps_ratio(text)
            txt_excl = count_exclamations(text)
            txt_sent = get_sentiment(text)
            txt_skew, txt_kurt = get_statistical_moments(text)
            lex_div, avg_word = get_lexical_features(text)
            
            # --- 2. ARRAY'I OLUŞTUR (Sıralama Eğitimdekiyle AYNI olmalı) ---
            # Sıra: [title_caps, title_excl, title_sent, text_caps, text_excl, text_sent, text_skew, text_kurt, lex_div, avg_word]
            
            num_features = np.array([[
                t_caps, t_excl, t_sent,
                txt_caps, txt_excl, txt_sent,
                txt_skew, txt_kurt,
                lex_div, avg_word
            ]])
            
            # --- 3. TF-IDF İLE BİRLEŞTİR ---
            combined_txt = title + " " + text
            cleaned_txt = clean_text_final(combined_txt)
            text_vector = tfidf_vectorizer.transform([cleaned_txt])
            
            final_features = hstack([text_vector, num_features])
            
            # Tahmin
            prediction = model.predict(final_features)[0]
            probs = model.predict_proba(final_features)[0]
            
            insights = generate_forensic_insights(title, text, num_features)
            
            self.after(100, lambda: self.show_results(prediction, probs, insights, title))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Failed", str(e)))
            self.reset_btn()

    def show_results(self, pred, probs, insights, title):
        fake_score = probs[1]
        
        if pred == 1:
            self.verdict_box.configure(fg_color="#4a1919")
            self.verdict_label.configure(text="FAKE NEWS DETECTED", text_color="#ff5555")
            self.verdict_icon.configure(text="🚫")
            self.score_label.configure(text=f"Confidence: {fake_score*100:.1f}%")
            self.add_to_history(title, True)
        else:
            self.verdict_box.configure(fg_color="#1a4a2e")
            self.verdict_label.configure(text="LIKELY REAL NEWS", text_color="#55ff99")
            self.verdict_icon.configure(text="✅")
            self.score_label.configure(text=f"Confidence: {probs[0]*100:.1f}%")
            self.add_to_history(title, False)

        self.insight_text.delete("0.0", "end")
        for line in insights:
            self.insight_text.insert("end", line + "\n\n")

        self.reset_btn()

    def reset_btn(self):
        self.after(0, lambda: self.analyze_btn.configure(state="normal", text="🔍 ANALYZE NEWS"))

if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()