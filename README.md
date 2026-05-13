# J.A.R.V.I.S - An AI Speech-to-Speech ChatBot Assistant

## Follow these steps to run J.A.R.V.I.S

### Step 1: Create a venv
```
python -m venv jarvis-env
```

### Step 2: Activate it
for Windows
```
jarvis-env\Scripts\activate
```
for Linux
```
source jarvis-env/bin/activate
```

### Step 3: Upgrade pip
```
pip install --upgrade pip
```

### Step 4.1: Install all dependencies
```
python -m pip install groq edge-tts pygame sounddevice soundfile numpy
```

### Step 4.2: Install .env
```
python -m pip install python-dotenv
```

### Step 5: Set your Groq API key
```
GROQ_API_KEY=your_key_here
```

### Step 6: Run the chatbot
```
python jarvis.py
```
