﻿## 🏥 Hospital Assistant – AI-Powered Query & Explanation System

> Built with 🧠 LLaMA3-70B + Groq + Supabase + Streamlit

---

### 📌 Overview

**Hospital Assistant** is an AI-powered tool that helps hospital staff ask **natural language questions** about **medical equipment**, **staff activity**, and **equipment locations or statuses**. It uses **Groq's ultra-fast LLaMA3-70B model** to generate smart SQL queries and provide simple, plain-English answers, backed by a **live Supabase PostgreSQL database**.

---

### 🎯 Objective

> The goal of this project is to allow hospital staff to **easily ask questions about equipment** – such as *“Where is the ventilator?”*, *“Who used the ECG machine last?”*, or *“What does a defibrillator do?”* – and get **accurate, easy-to-understand answers instantly**. Whether it’s about usage, location, status, or purpose of medical devices, staff can now interact with the hospital’s database using **just plain English** – no technical knowledge required.

---

### 🔥 Quick Capabilities

* 🔍 Ask: "Where is the ventilator?" or "Who used the ECG machine last?"
* 🧠 Understand: "What is a defibrillator?" or "What does an MRI do?"
* 🚦 Check: Equipment status (e.g., available, in use, under maintenance)
* 🔄 Auto-retry: Failed queries are retried for robustness

This is a **zero-training, conversational interface** for real-time hospital database access.

---

### 💡 Key Features

| Feature                     | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| 🧠 LLM-based SQL Generation | Uses LLaMA3-70B (via Groq API) to convert plain English to SQL queries |
| 🔄 Auto Query Retry         | Retries SQL execution automatically if Supabase fails                  |
| 🩺 Medical Explanations     | Explains equipment usage/purpose in simple terms                       |
| 🧵 Multi-Question Handling  | Handles multiple questions in one message                              |
| 🔧 Supabase RPC Support     | Uses Supabase's `execute_sql()` function for secure query execution    |
| 💬 Streamlit UI             | Chat-style interface with conversational memory                        |
| 🤖 Dual-Agent System        | Agent 1 generates SQL/explanations, Agent 2 summarizes responses       |

---

### 🧱 Tech Stack

| Layer         | Tools Used                                               |
| ------------- | -------------------------------------------------------- |
| 💬 LLM Engine | [Groq API](https://console.groq.com) with LLaMA3-70B     |
| 📦 Database   | [Supabase](https://supabase.com) – PostgreSQL with RPC   |
| 💻 Frontend   | [Streamlit](https://streamlit.io) for the user interface |
| 🧠 Agents     | Custom system prompts + tool calling                     |
| 🛠️ Backend   | Python, dotenv, logging, retry logic                     |

---

### 🗂️ Database Schema

A real-world hospital data structure with interconnected tables:

#### 📊 Tables

| Table         | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `departments` | Hospital departments (ICU, Surgery, Radiology, etc.)                   |
| `employees`   | Doctors, nurses, and hospital staff                                    |
| `equipment`   | Medical equipment with model, type, status, and location details       |
| `locations`   | Building, floor, room coordinates for where equipment is placed        |
| `usage_logs`  | Tracks employee interactions with equipment (who used what, when, why) |

#### 🔗 Relationships

* **employees → departments**
* **equipment → departments + locations + last used by (employee)**
* **usage\_logs → equipment + employees**

Example flow:

```
employees → departments  
equipment → locations + departments  
usage_logs → equipment + employees  
```

---

### 🧠 Agent System

#### 🧠 Agent 1: Query Generator

* Detects intent (data request vs explanation)
* Generates SQL queries when needed
* Handles multi-question prompts
* Responds to definition-based questions with explanations

#### 🧠 Agent 2: Response Explainer

* Converts raw data results into natural language
* No SQL, no table names – just human-friendly responses
* Handles errors and empty results gracefully

---

### 🔄 Retry Mechanism

Both query generation and execution include built-in retry logic:

* Retries up to **2 times** if there's an error
* Adds a short **2-second delay** between retries
* Makes the app **resilient to transient Supabase/API issues**

---

Let's add a 🔊 **Voice Interaction** section to your README just below the **📌 Sample Queries** part. This will showcase the **Groq TTS integration**, audio input, and how hospital staff can use the app hands-free. Here’s the markdown you can directly paste:

---

### 🔊 Voice Interaction Support

With integrated **voice input and response**, staff can now **speak their questions** and **hear back answers**, powered by:

* 🎙️ **Microphone-based input** (via `audio_recorder_streamlit`)
* 🧠 Real-time transcription and intent detection
* 🗣️ **Groq's `playai-tts` models** for fast, natural voice replies
* 🔁 Seamless switch between text and voice modes

#### ✅ How It Works:

1. Click the **🎙️ microphone button** to ask your question out loud.
2. The assistant transcribes your speech using a Whisper model or `transcribe_audio()` logic.
3. It interprets the query (SQL or explanation) just like a typed input.
4. The answer is spoken back using your selected voice (e.g., Judy, Mitch, Celeste via `playai-tts`).

#### 🎭 Voice Options

Choose from multiple AI-generated voices in the sidebar:

* Judy-PlayAI (Neutral, default)
* Mitch-PlayAI (Warm)
* Celeste-PlayAI (Calm)
* Chip-PlayAI (Upbeat)
* Jennifer-PlayAI (Professional)
* Basil-PlayAI (Sharp)

#### 💡 Why It Matters

---

### 🔒 Database Access Note

> ⚠️ **Security Best Practice**:
> The model currently has full access to the database for development. However, it's **strongly recommended** to assign **read-only access** via Supabase roles to ensure safety and prevent unwanted modifications. You can do this by limiting permissions for the role tied to the API key used.

---

### ▶️ Run Locally

#### 1. Clone the repo

```bash
git clone https://github.com/yourusername/hospital-assistant-ai.git
cd hospital-assistant-ai
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set up environment variables

Create a `.env` file with your keys:

```env
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key
```

#### 4. Set up Supabase DB

* Use the schema provided above
* Add a stored procedure: `execute_sql(query TEXT)`
* Ensure proper permissions (preferably **read-only**)

#### 5. Run the app

```bash
streamlit run app.py
```

---

### 📌 Sample Queries You Can Ask

* “Where is the ECG machine?”
* “What is a defibrillator?”
* “Who last used the ventilator?”
* “Is the MRI machine currently in use?”
* “Where is the ventilator and who used it last?”

---
