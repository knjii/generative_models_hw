"""
app_v4.py — Версия 4: Function Calling — ИИ сам решает когда генерировать

Что добавилось по сравнению с v3:
  - Убрана отдельная кнопка "Нарисуй"
  - Модель сама анализирует запрос и решает: ответить текстом ИЛИ вызвать generate_image
  - Два запроса к API: 1-й — с tools, 2-й — получить итоговый ответ

Стек: Flask + OpenAI-совместимый API (chat: gpt-4o-mini, генерация: gpt-image-1)
Запуск: python app_v4.py
Открой: http://localhost:5004
"""

import os
import json
import base64
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI

from dotenv import load_dotenv


if load_dotenv:
    load_dotenv()

API_KEY  = (
    os.environ.get("API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("BOTHUB_API_KEY", "")
)
BASE_URL = (
    os.environ.get("OPENROUTER_BASE_URL")
    or os.environ.get("BOTHUB_API_LINK")
    or "https://openrouter.ai/api/v1"
)
OPENROUTER_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:5004")
OPENROUTER_TITLE   = os.environ.get("OPENROUTER_APP_TITLE", "Flask Function Calling Demo")
DEFAULT_HEADERS    = {
    "HTTP-Referer": OPENROUTER_REFERER,
    "X-OpenRouter-Title": OPENROUTER_TITLE,
}

CHAT_MODEL    = os.environ.get("CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
IMAGE_MODEL   = "gpt-image-1"    # модель генерации изображений
IMAGE_SIZE    = "1024x1024"      # размер генерируемого изображения
SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай кратко. "
    "Когда пользователь просит нарисовать, создать, визуализировать "
    "или сгенерировать любое изображение — используй инструмент generate_image. "
    "Когда пользователь спрашивает текущую дату или время — используй "
    "инструмент get_current_datetime."
)
PORT = 5004

FALLBACK_TIMEZONES = {
    "UTC": 0,
    "Europe/Moscow": 3,
    "Europe/London": 0,
    "Europe/Berlin": 1,
    "America/New_York": -5,
    "Asia/Tokyo": 9,
}

# ─── Описание инструментов (JSON-схемы для модели) ───────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generates an image from a text description. "
                "Call this when the user wants to draw, create, visualize, "
                "or generate any visual content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed image description in English. "
                            "Be specific about style, composition, colors."
                        )
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Returns the current date and time for a requested timezone. "
                "Call this when the user asks what time or date it is now."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone name, for example Europe/Moscow, "
                            "Europe/London, America/New_York. "
                            "Use Europe/Moscow if the user does not specify one."
                        )
                    }
                },
                "required": []
            }
        }
    }
]

# ═══════════════════════════════════════════════════════════════

app     = Flask(__name__)
client  = OpenAI(api_key=API_KEY, base_url=BASE_URL, default_headers=DEFAULT_HEADERS)
history = []

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Чат v4 — Function Calling</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b;
           display: flex; flex-direction: column; height: 100vh; }
    header { padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #e2e8f0;
             font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 10px; }
    .tag { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
           background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
            flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 12px;
           line-height: 1.6; font-size: 15px; white-space: pre-wrap; }
    .msg.user      { align-self: flex-end; background: #2563eb; color: white; }
    .msg.assistant { align-self: flex-start; background: #ffffff; color: #1e293b;
                     border: 1px solid #e2e8f0; box-shadow: 0 1px 3px #0001; }
    .msg.system    { align-self: center; font-size: 12px; color: #94a3b8; font-style: italic; }
    .msg.error     { align-self: center; background: #fee2e2; color: #dc2626;
                     border: 1px solid #fca5a5; font-size: 13px; }
    .msg.tool-call { align-self: flex-start; background: #fffbeb; color: #92400e;
                     border: 1px solid #fde68a; font-size: 12px; font-family: monospace; }
    .msg.image-result { align-self: flex-start; background: transparent; padding: 0; }
    .msg.image-result img { max-width: 380px; border-radius: 12px;
                             border: 2px solid #e2e8f0; display: block;
                             box-shadow: 0 4px 12px #0002; }
    .msg img.inline { max-width: 220px; border-radius: 8px; display: block; margin-bottom: 6px; }
    #form { padding: 12px 20px 16px; background: #ffffff; border-top: 1px solid #e2e8f0; }
    #preview-area { display: none; align-items: center; gap: 8px; margin-bottom: 10px;
                    padding: 8px 10px; background: #f1f5f9; border-radius: 8px; }
    #preview-area img { max-height: 80px; border-radius: 6px; border: 1px solid #e2e8f0; }
    #preview-area button { background: none; border: none; color: #94a3b8;
                           cursor: pointer; font-size: 18px; }
    #inputs { display: flex; gap: 8px; }
    #file-label { padding: 10px 12px; border-radius: 8px; background: #f1f5f9; color: #64748b;
                  cursor: pointer; font-size: 18px; border: 1px solid #e2e8f0;
                  display: flex; align-items: center; }
    #file-label:hover { background: #e2e8f0; }
    #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1;
             background: #f8fafc; color: #1e293b; font-size: 15px; outline: none;
             resize: none; height: 44px; }
    #input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px #dbeafe; }
    .btn { padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer;
           font-size: 14px; font-weight: 600; transition: opacity .2s; }
    .btn:hover    { opacity: 0.85; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-send  { background: #2563eb; color: white; }
    .btn-reset { background: #f1f5f9; color: #64748b; }
  </style>
</head>
<body>
  <header>
    🤖 Чат-бот
    <span class="tag">v4 — Function Calling</span>
  </header>

  <div id="chat">
    <div class="msg system">
      Умный чат: просите нарисовать что-нибудь — модель сама решит вызвать генерацию.<br>
      Например: «Нарисуй закат над горами» или «Создай логотип для кафе»
    </div>
  </div>

  <div id="form">
    <div id="preview-area">
      <img id="preview-img" src="" alt="">
      <span style="font-size:13px;color:#64748b;flex:1" id="preview-name"></span>
      <button onclick="clearImage()">✕</button>
    </div>
    <div id="inputs">
      <label id="file-label" for="file-input" title="Прикрепить изображение">📎</label>
      <input id="file-input" type="file" accept="image/*" style="display:none">
      <textarea id="input" placeholder='Напишите сообщение. Например: "Нарисуй кота-астронавта"'></textarea>
      <button class="btn btn-send"  id="send-btn">Отправить</button>
      <button class="btn btn-reset" id="reset-btn">↺</button>
    </div>
  </div>

  <script>
    const chat    = document.getElementById('chat');
    const input   = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');
    let selectedFile = null;

    document.getElementById('file-input').onchange = (e) => {
      selectedFile = e.target.files[0];
      if (!selectedFile) return;
      document.getElementById('preview-img').src = URL.createObjectURL(selectedFile);
      document.getElementById('preview-name').textContent = selectedFile.name;
      document.getElementById('preview-area').style.display = 'flex';
    };

    function clearImage() {
      selectedFile = null;
      document.getElementById('file-input').value = '';
      document.getElementById('preview-area').style.display = 'none';
    }

    function addMsg(text, role) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const text = input.value.trim();
      if (!text && !selectedFile) return;

      const formData = new FormData();
      if (text)         formData.append('message', text);
      if (selectedFile) formData.append('image', selectedFile);

      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      if (selectedFile) {
        const img = document.createElement('img');
        img.className = 'inline';
        img.src = URL.createObjectURL(selectedFile);
        userDiv.appendChild(img);
      }
      if (text) userDiv.appendChild(document.createTextNode(text));
      chat.appendChild(userDiv);
      chat.scrollTop = chat.scrollHeight;

      input.value = '';
      clearImage();
      sendBtn.disabled = true;

      const thinking = addMsg('🤔 Думаю...', 'assistant');

      try {
        const res  = await fetch('/chat', { method: 'POST', body: formData });
        const data = await res.json();

        thinking.remove();

        if (!res.ok || data.error) {
          addMsg('Ошибка: ' + (data.error || res.status), 'error');
          return;
        }

        // Если были вызваны функции — показываем детали вызовов
        if (data.tool_calls && data.tool_calls.length) {
          for (const tool of data.tool_calls) {
            addMsg(
              'Вызов инструмента: ' + tool.name + '\\nАргументы: ' +
              JSON.stringify(tool.arguments, null, 2),
              'tool-call'
            );
          }
        }

        // Показываем сгенерированное изображение
        if (data.image_url) {
          const imgDiv = document.createElement('div');
          imgDiv.className = 'msg image-result';
          const img = document.createElement('img');
          img.src = data.image_url;
          img.alt = 'Сгенерированное изображение';
          imgDiv.appendChild(img);
          chat.appendChild(imgDiv);
          chat.scrollTop = chat.scrollHeight;
        }

        // Текстовый ответ модели
        if (data.reply) addMsg(data.reply, 'assistant');

      } catch (e) {
        thinking.remove();
        addMsg('Ошибка: ' + e.message, 'error');
      } finally {
        sendBtn.disabled = false;
        input.focus();
      }
    }

    sendBtn.onclick = sendMessage;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    document.getElementById('reset-btn').onclick = async () => {
      await fetch('/reset', { method: 'POST' });
      chat.innerHTML = '<div class="msg system">История сброшена.</div>';
    };
  </script>
</body>
</html>
"""

# ─── Вспомогательные функции, которые может вызвать модель ───────────────────

def do_generate_image(prompt: str) -> str:
    """Вызывает gpt-image-1, возвращает изображение как data URL (base64)."""
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        n=1
    )
    # gpt-image-1 возвращает base64 (b64_json), не URL
    b64 = response.data[0].b64_json
    return f"data:image/png;base64,{b64}"


def do_get_current_datetime(timezone_name: str = "Europe/Moscow") -> str:
    """Возвращает текущую дату и время в выбранном часовом поясе."""
    timezone_name = timezone_name or "Europe/Moscow"
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        offset_hours = FALLBACK_TIMEZONES.get(timezone_name)
        if offset_hours is None:
            timezone_name = "Europe/Moscow"
            offset_hours = FALLBACK_TIMEZONES[timezone_name]
        tz = timezone(timedelta(hours=offset_hours), timezone_name)

    now = datetime.now(tz)
    return json.dumps(
        {
            "timezone": timezone_name,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "utc_offset": now.strftime("%z"),
        },
        ensure_ascii=False
    )


# ─── Маршруты ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('message', '').strip()
        image_file   = request.files.get('image')

        # ── Формируем content (текст или текст+картинка) ──────────────────
        if image_file and image_file.filename:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type  = image_file.content_type or 'image/jpeg'
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": user_message or "Опиши картинку"}
            ]
            history_text = f"[картинка] {user_message}" if user_message else "[картинка]"
        else:
            if not user_message:
                return jsonify({"error": "Пустое сообщение"}), 400
            content      = user_message
            history_text = user_message

        history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        # ── Первый запрос: передаём tools, модель может вернуть tool_call ──
        resp1 = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS
        )

        msg        = resp1.choices[0].message
        image_url  = None
        tool_calls_info = []

        # ── Проверяем: хочет ли модель вызвать функцию? ───────────────────
        if msg.tool_calls:
            # Добавляем в messages: ответ ассистента + результат tool
            messages.append({
                "role":       "assistant",
                "content":    None,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if tool_name == "generate_image":
                    prompt = args.get("prompt", "")
                    image_url = do_generate_image(prompt)
                    tool_result = "Image generated successfully."
                elif tool_name == "get_current_datetime":
                    tool_result = do_get_current_datetime(
                        args.get("timezone", "Europe/Moscow")
                    )
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                tool_calls_info.append({
                    "name": tool_name,
                    "arguments": args,
                    "result": tool_result
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      tool_result
                })

            # ── Второй запрос: получаем финальный текстовый ответ ─────────
            resp2 = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS
            )
            reply = resp2.choices[0].message.content
        else:
            # Нет tool_call — обычный текстовый ответ
            reply = msg.content

        # ── Обновляем историю (без base64 чтобы не раздувать контекст) ────
        history[-1] = {"role": "user", "content": history_text}
        history.append({"role": "assistant", "content": reply or ""})

        result = {
            "reply": reply,
            "tool_called": bool(tool_calls_info),
            "tool_calls": tool_calls_info
        }
        if image_url:
            result["image_url"] = image_url
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    history.clear()
    return jsonify({"status": "ok"})


# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print(f"Чат-бот v4 — Function Calling  |  http://localhost:{PORT}")
    print(f"Модель: {CHAT_MODEL}")
    print(f"API: {BASE_URL}")
    print("=" * 50)
    app.run(debug=True, port=PORT)
