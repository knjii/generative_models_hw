"""
arena.py — Мини LM Arena
════════════════════════════════════════════════════════════════
Учебный проект: сравниваем ответы двух локальных моделей,
пользователь голосует за лучший, система ведёт Elo-рейтинг.

Ключевые особенности:
  - Определяет установленные Ollama-модели через локальный API
  - Показывает пошаговые вычисления Elo после каждого голоса
  - Кнопка автоматической генерации вопроса
  - Слепое тестирование (имена моделей скрыты до голосования)

Требования:
  - Ollama запущена локально (порт 11434)
  - Установлено минимум 2 chat-модели: ollama pull <model>
  - pip install flask

Запуск: python arena.py  →  http://localhost:5010
"""

import argparse
import json
import os
import random
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, jsonify, render_template_string

# ********************* КОНФИГУРАЦИЯ *********************
# Все настраиваемые константы вынесены сюда

OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_URL      = os.environ.get("OLLAMA_OPENAI_URL", f"{OLLAMA_HOST}/v1")
OLLAMA_CHAT_API = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_API = f"{OLLAMA_HOST}/api/tags"
OLLAMA_THINK    = os.environ.get("OLLAMA_THINK", "false").lower() in ("1", "true", "yes", "on")
ARENA_MODELS    = [
    m.strip()
    for m in os.environ.get("ARENA_MODELS", "").split(",")
    if m.strip()
]                                                # опционально: ARENA_MODELS=qwen3.5:4b,ministral-3:8b
PORT            = 5010                           # порт нашего Flask-сервера
MAX_TOKENS      = 1024                           # максимум токенов в ответе модели
TEMPERATURE     = 0.7                            # температура генерации (0=детерм., 1=творч.)
ELO_K           = 32                             # коэффициент K: насколько сильно меняется рейтинг за одну битву
ELO_START       = 1000                           # начальный Elo для каждой новой модели
EXPORT_PATH     = Path("arena_tests.json")       # куда сохраняем историю тестов

# ********************* ИНИЦИАЛИЗАЦИЯ *********************

app    = Flask(__name__)

# Словарь рейтингов: { model_id: {"elo": float, "wins": int, "losses": int, "ties": int, "battles": int} }
ratings = {}

# Журнал тестов для экспорта: вопросы, ответы, голоса и последовательные изменения Elo.
test_history = []

# Временное хранилище боёв между /ask и /vote.
pending_battles = {}

history_lock = threading.Lock()


# ********************* ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ *********************

def get_loaded_models():
    """
    Возвращает список моделей Ollama, доступных для chat completions.

    Ollama не требует отдельной кнопки Load: если модель установлена локально,
    сервер загрузит её в память при первом запросе. Поэтому используем
    GET /api/tags и исключаем embedding-модели.

    Если задана переменная окружения ARENA_MODELS, используем только её:
      ARENA_MODELS=qwen3.5:4b,ministral-3:8b
    """
    if ARENA_MODELS:
        print(f"[get_loaded_models] Модели из ARENA_MODELS: {len(ARENA_MODELS)}")
        for mid in ARENA_MODELS:
            print(f"  - {mid}")
        return ARENA_MODELS

    try:
        with urllib.request.urlopen(OLLAMA_TAGS_API, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # Выводим ошибку в консоль — помогает диагностировать проблему
        print(f"[get_loaded_models] Ошибка запроса к {OLLAMA_TAGS_API}: {e}")
        print("  Убедитесь что Ollama запущена: ollama serve")
        return []

    # Структура ответа Ollama /api/tags:
    # {
    #   "models": [
    #     {
    #       "name": "qwen3.5:4b",
    #       "model": "qwen3.5:4b",
    #       "details": {"family": "qwen35", ...}
    #     }, ...
    #   ]
    # }
    all_models = data.get("models", [])

    available = []
    for m in all_models:
        model_id = m.get("model") or m.get("name")
        if not model_id:
            continue
        details = m.get("details", {})
        searchable = " ".join([
            model_id,
            details.get("family", "") or "",
            " ".join(details.get("families", []) or []),
        ]).lower()
        if "embed" in searchable or "embedding" in searchable:
            continue
        available.append(model_id)

    print(f"[get_loaded_models] Всего моделей Ollama: {len(all_models)}, chat-кандидатов: {len(available)}")
    for mid in available:
        print(f"  - {mid}")
    if not available:
        print("  Не найдено chat-моделей. Установите минимум две: ollama pull <model>")
    return available


def ensure_rating(model_id):
    """
    Инициализирует запись рейтинга для модели, если её ещё нет в словаре.
    Идемпотентна: безопасно вызывать многократно.
    """
    if model_id not in ratings:
        ratings[model_id] = {
            "elo":     float(ELO_START),  # стартовый рейтинг
            "wins":    0,                  # победы
            "losses":  0,                  # поражения
            "ties":    0,                  # ничьи
            "battles": 0                   # всего битв (wins + losses + ties)
        }


def now_iso():
    """Текущее время в ISO-формате для JSON-журнала."""
    return datetime.now(timezone.utc).isoformat()


def snapshot_ratings():
    """Возвращает копию рейтингов, безопасную для записи в JSON."""
    return {
        model_id: {
            "elo": data["elo"],
            "wins": data["wins"],
            "losses": data["losses"],
            "ties": data["ties"],
            "battles": data["battles"],
        }
        for model_id, data in ratings.items()
    }


def save_test_history(path=None):
    """Экспортирует накопленную историю тестов и текущие рейтинги в JSON."""
    path = path or EXPORT_PATH
    payload = {
        "schema_version": 1,
        "exported_at": now_iso(),
        "settings": {
            "elo_start": ELO_START,
            "elo_k": ELO_K,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "total_tests": len(test_history),
        "ratings": snapshot_ratings(),
        "tests": test_history,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def apply_vote(left, right, winner):
    """
    Применяет голос к рейтингам и возвращает снимки до/после.
    winner: "left", "right" или "tie".
    """
    for m in [left, right]:
        ensure_rating(m)

    before = snapshot_ratings()

    for m in [left, right]:
        ratings[m]["battles"] += 1

    elo_details = None
    if winner == "left":
        elo_details = elo_update(left, right)
        ratings[left]["wins"] += 1
        ratings[right]["losses"] += 1
    elif winner == "right":
        elo_details = elo_update(right, left)
        ratings[right]["wins"] += 1
        ratings[left]["losses"] += 1
    else:
        ratings[left]["ties"] += 1
        ratings[right]["ties"] += 1

    after = snapshot_ratings()
    return before, after, elo_details


def build_rating_corrections(left, right, before, after, winner):
    """Формирует понятный список последовательных изменений рейтинга."""
    corrections = []
    for model_id, side in [(left, "left"), (right, "right")]:
        result = "tie"
        if winner == side:
            result = "win"
        elif winner in ("left", "right"):
            result = "loss"

        corrections.append({
            "model": model_id,
            "side": side,
            "result": result,
            "elo_before": before[model_id]["elo"],
            "elo_after": after[model_id]["elo"],
            "elo_delta": round(after[model_id]["elo"] - before[model_id]["elo"], 1),
            "stats_before": before[model_id],
            "stats_after": after[model_id],
        })
    return corrections


def record_test(question, left, right, ans_left, ans_right, winner, before, after,
                elo_details, battle_id=None):
    """Добавляет завершённый бой в журнал и сразу сохраняет JSON на диск."""
    record = {
        "test_number": len(test_history) + 1,
        "battle_id": battle_id or str(uuid4()),
        "completed_at": now_iso(),
        "question": question,
        "models": {
            "left": left,
            "right": right,
        },
        "answers": {
            "left": ans_left,
            "right": ans_right,
        },
        "vote": {
            "winner": winner,
            "winner_model": left if winner == "left" else right if winner == "right" else None,
            "is_tie": winner == "tie",
        },
        "ratings_before": before,
        "elo_details": elo_details,
        "rating_corrections": build_rating_corrections(left, right, before, after, winner),
        "ratings_after": after,
    }
    test_history.append(record)
    save_test_history()
    return record


def _call_model(model_id, messages, result, idx):
    """
    Запрашивает ответ у одной модели. Предназначена для запуска в потоке.
    result[idx] — куда записать текст ответа или сообщение об ошибке.

    Используем native Ollama API /api/chat, а не OpenAI-compatible /v1,
    потому что native API поддерживает think=False для reasoning-моделей.
    """
    short_name = model_id.split('/')[-1]  # короткое имя для логов
    try:
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "think": OLLAMA_THINK,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }
        req = urllib.request.Request(
            OLLAMA_CHAT_API,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        msg = data.get("message", {})
        content = msg.get("content") or msg.get("thinking")
        if not content:
            content = "[Модель вернула пустой ответ]"

        print(f"[_call_model] {short_name}: {len(content)} символов")
        result[idx] = content

    except Exception as e:
        # Фиксируем ошибку в лог и возвращаем читаемое сообщение
        print(f"[_call_model] {short_name}: ОШИБКА — {e}")
        result[idx] = f"[Ошибка модели {short_name}: {e}]"


def elo_update(winner_id, loser_id):
    """
    Пересчитывает Elo-рейтинг после победы winner над loser.

    Формула Elo:
      expected_w = 1 / (1 + 10^((Elo_loser - Elo_winner) / 400))
      new_winner = old_winner + K * (1 - expected_w)
      new_loser  = old_loser  + K * (0 - expected_w)   [эквивалентно: - K * (1 - expected_w)]

    Интуиция: если победитель был сильнее (ожидаемо выиграл) — очков мало.
    Если победитель был слабее (неожиданная победа) — очков много.

    Возвращает dict с детальной информацией о вычислении (для UI).
    """
    ew_before = ratings[winner_id]["elo"]  # рейтинг победителя ДО битвы
    el_before = ratings[loser_id]["elo"]   # рейтинг проигравшего ДО битвы

    # Ожидаемая вероятность победы: вероятность согласно текущим рейтингам
    expected_w = 1.0 / (1.0 + 10.0 ** ((el_before - ew_before) / 400.0))

    # Изменение рейтинга
    delta = round(ELO_K * (1 - expected_w), 1)

    # Обновляем рейтинги
    ratings[winner_id]["elo"] = round(ew_before + delta, 1)
    ratings[loser_id]["elo"]  = round(el_before  - delta, 1)

    # Возвращаем подробности для отображения в UI (учебная цель!)
    return {
        "winner":       winner_id,
        "loser":        loser_id,
        "ew_before":    ew_before,
        "el_before":    el_before,
        "expected_pct": round(expected_w * 100, 1),   # ожидаемая вероятность в %
        "delta":        delta,                          # сколько очков передаётся
        "ew_after":     ratings[winner_id]["elo"],
        "el_after":     ratings[loser_id]["elo"],
        "K":            ELO_K
    }


# ********************* FLASK МАРШРУТЫ *********************

@app.route("/")
def index():
    """Главная страница — отдаём встроенный HTML."""
    return render_template_string(HTML)


@app.route("/models")
def models_route():
    """
    GET /models — возвращает список установленных chat-моделей Ollama.
    """
    models = get_loaded_models()
    for m in models:
        ensure_rating(m)   # инициализируем рейтинг для каждой найденной модели
    return jsonify({"models": models})


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask  { "question": "..." }
    Выбирает 2 случайные модели Ollama, задаёт им вопрос параллельно.
    Возвращает ответы в случайном порядке (слепой тест).
    """
    data     = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Пустой вопрос"}), 400

    # Получаем доступные локальные модели Ollama
    available = get_loaded_models()
    for m in available:
        ensure_rating(m)

    if len(available) < 2:
        return jsonify({"error": "Нужно минимум 2 установленные chat-модели Ollama: ollama pull <model>"}), 400

    # Выбираем 2 случайные УНИКАЛЬНЫЕ модели
    model_a, model_b = random.sample(available, 2)
    messages = [{"role": "user", "content": question}]

    # Контейнер для результатов потоков: out[0] = ответ model_a, out[1] = ответ model_b
    out = [None, None]

    # Запускаем оба запроса параллельно — экономим время ожидания
    t1 = threading.Thread(target=_call_model, args=(model_a, messages, out, 0))
    t2 = threading.Thread(target=_call_model, args=(model_b, messages, out, 1))
    t1.start(); t2.start()
    t1.join();  t2.join()  # ждём завершения ОБОИХ потоков

    # Случайный порядок отображения (слепой тест: пользователь не знает кто слева)
    if random.random() < 0.5:
        left, right = model_a, model_b
        ans_left, ans_right = out[0], out[1]
    else:
        left, right = model_b, model_a
        ans_left, ans_right = out[1], out[0]

    battle_id = str(uuid4())
    pending_battles[battle_id] = {
        "created_at": now_iso(),
        "question": question,
        "left": left,
        "right": right,
        "ans_left": ans_left,
        "ans_right": ans_right,
    }

    return jsonify({
        "battle_id":  battle_id,
        "question":   question,
        "left":      left,
        "right":     right,
        "ans_left":  ans_left,
        "ans_right": ans_right
    })


@app.route("/autoquestion", methods=["POST"])
def autoquestion():
    """
    POST /autoquestion — генерирует случайный тестовый вопрос с помощью модели.
    Выбирает случайную модель Ollama и просит её придумать интересный вопрос.
    Промпт включает 10 примеров, чтобы ИИ понял стиль нужных вопросов.
    """
    available = get_loaded_models()
    if not available:
        return jsonify({"error": "Нет установленных chat-моделей Ollama"}), 400

    # Берём случайную модель-генератор вопроса
    generator = random.choice(available)

    # Системный промпт с примерами — ИИ поймёт что мы хотим
    system_prompt = (
        "Ты генератор тестовых вопросов для сравнения языковых моделей. "
        "Придумай ОДИН интересный вопрос, на который разные модели дадут "
        "заметно разные ответы. Отвечай ТОЛЬКО вопросом, без предисловий.\n\n"
        "Примеры хороших вопросов:\n"
        "- Напиши функцию на Python для поиска всех простых чисел до N методом решета Эратосфена\n"
        "- Объясни разницу между TCP и UDP простыми словами за 3 предложения\n"
        "- В чём плюсы и минусы микросервисной архитектуры по сравнению с монолитом?\n"
        "- Напиши SQL-запрос: топ-5 пользователей по сумме заказов за последний месяц\n"
        "- Что такое переобучение нейросети и как с ним бороться?\n"
        "- Объясни квантовую запутанность так, чтобы понял школьник\n"
        "- Чем отличается сортировка слиянием от быстрой сортировки? Когда что использовать?\n"
        "- Напиши регулярное выражение для проверки email-адреса и объясни его\n"
        "- Что такое CAP-теорема и как она влияет на выбор базы данных?\n"
        "- Какие существуют паттерны для обработки ошибок в асинхронном Python коде?\n"
    )

    result = [None]
    _call_model(generator, [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Придумай новый вопрос в том же стиле."}
    ], result, 0)

    question = result[0] or "Объясни разницу между процессом и потоком в операционной системе."
    # Убираем кавычки и лишние символы если модель добавила их
    question = question.strip().strip('"').strip("'").strip()

    return jsonify({"question": question, "generator": generator})


@app.route("/vote", methods=["POST"])
def vote():
    """
    POST /vote  { "winner": "left"|"right"|"tie", "left": model_id, "right": model_id }
    Фиксирует голос, пересчитывает Elo, возвращает подробности вычисления.
    """
    data   = request.get_json()
    winner = data.get("winner")   # кто победил: "left", "right" или "tie"
    left   = data.get("left")     # id модели слева
    right  = data.get("right")    # id модели справа
    battle_id = data.get("battle_id")

    if winner not in ("left", "right", "tie"):
        return jsonify({"error": "winner должен быть left, right или tie"}), 400
    if not left or not right:
        return jsonify({"error": "Нужны left и right"}), 400

    with history_lock:
        before, after, elo_details = apply_vote(left, right, winner)
        battle = pending_battles.pop(battle_id, {}) if battle_id else {}
        test_record = record_test(
            question=battle.get("question", data.get("question", "")),
            left=left,
            right=right,
            ans_left=battle.get("ans_left", data.get("ans_left", "")),
            ans_right=battle.get("ans_right", data.get("ans_right", "")),
            winner=winner,
            before=before,
            after=after,
            elo_details=elo_details,
            battle_id=battle_id,
        )

    # Возвращаем и текущий рейтинг, и детали вычисления для отображения в UI
    return jsonify({
        "ratings": ratings,
        "elo_details": elo_details,
        "test_record": test_record,
        "export_path": str(EXPORT_PATH)
    })


@app.route("/stats")
def stats():
    """GET /stats — текущая рейтинговая таблица всех моделей."""
    return jsonify({"ratings": ratings})


@app.route("/export")
def export_route():
    """GET /export — отдаёт JSON с историей тестов и текущими рейтингами."""
    with history_lock:
        payload = save_test_history()
    return jsonify(payload)


# ********************* HTML / JS ФРОНТЕНД *********************
# Весь интерфейс встроен в Python-строку (паттерн render_template_string).
# Разделы HTML:
#   1. <style>  — CSS-стили
#   2. <body>   — разметка: шапка, блок вопроса, арена, Elo-панель, статистика
#   3. <script> — логика: запросы к API, рендер карточек, Elo-визуализация, Chart.js

HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мини LM Arena</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  /* ── Переменные цветов ── */
  :root {
    --bg: #f1f5f9; --card: #ffffff; --border: #e2e8f0;
    --text: #1e293b; --muted: #64748b;
    --blue: #3b82f6; --green: #22c55e; --yellow: #f59e0b;
    --red: #ef4444;  --purple: #8b5cf6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; min-height: 100vh; }

  /* ── Шапка ── */
  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 17px; font-weight: 700; flex: 1; }
  .badge { font-size: 12px; color: var(--muted); background: #f1f5f9; padding: 3px 10px; border-radius: 20px; white-space: nowrap; }

  /* ── Основной контейнер ── */
  .main { max-width: 980px; margin: 0 auto; padding: 18px 16px; display: flex; flex-direction: column; gap: 14px; }

  /* ── Карточка-секция ── */
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .section-title { font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; margin-bottom: 10px; }

  /* ── Блок вопроса ── */
  .ask-row { display: flex; gap: 8px; align-items: flex-end; }
  .ask-row textarea { flex: 1; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; font-size: 14px; resize: vertical; min-height: 68px; font-family: inherit; color: var(--text); background: var(--bg); }
  .ask-row textarea:focus { outline: none; border-color: var(--blue); }
  .ask-btns { display: flex; flex-direction: column; gap: 6px; }
  .btn { padding: 9px 18px; border: none; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity .2s; white-space: nowrap; }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--blue); color: #fff; }
  .btn-primary:hover:not(:disabled) { opacity: .85; }
  .btn-auto { background: var(--purple); color: #fff; }
  .btn-auto:hover:not(:disabled) { opacity: .85; }

  /* ── Арена: две карточки ── */
  .arena { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .card { background: var(--card); border: 2px solid var(--border); border-radius: 12px; padding: 14px; display: flex; flex-direction: column; gap: 8px; min-height: 180px; transition: border-color .25s, box-shadow .25s; }
  .card.winner { border-color: var(--green); box-shadow: 0 0 0 3px #22c55e22; }
  .card.loser  { opacity: .75; }
  .card-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .card-answer { font-size: 13.5px; line-height: 1.75; color: var(--text); flex: 1; white-space: pre-wrap; overflow-y: auto; max-height: 320px; }
  .card-model { font-size: 11px; color: var(--blue); font-style: italic; display: none; padding: 4px 0 0; border-top: 1px solid var(--border); }

  /* ── Кнопки голосования ── */
  .vote-row { display: flex; gap: 10px; justify-content: center; padding: 4px 0; }
  .btn-vote { padding: 10px 26px; border: 2px solid var(--border); background: var(--card); border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all .2s; }
  .btn-vote:hover:not(:disabled) { transform: translateY(-1px); }
  .btn-vote.left:hover:not(:disabled)  { border-color: var(--blue);   color: var(--blue); }
  .btn-vote.right:hover:not(:disabled) { border-color: var(--yellow);  color: var(--yellow); }
  .btn-vote.tie:hover:not(:disabled)   { border-color: var(--muted);   color: var(--muted); }
  .btn-vote:disabled { opacity: .4; cursor: not-allowed; }

  /* ── Панель Elo-вычислений ── */
  .elo-panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; color: #e2e8f0; font-family: 'Courier New', monospace; font-size: 13px; line-height: 2; }
  .elo-panel .ep-title { font-family: system-ui, sans-serif; font-size: 13px; font-weight: 700; color: #94a3b8; margin-bottom: 10px; text-transform: uppercase; letter-spacing: .4px; }
  .elo-step { display: flex; align-items: baseline; gap: 8px; }
  .elo-step .lbl { color: #475569; min-width: 200px; }
  .elo-step .val { color: #f8fafc; font-weight: 700; }
  .elo-step .val.up   { color: #4ade80; }
  .elo-step .val.down { color: #f87171; }
  .elo-step .val.eq   { color: #94a3b8; }
  .elo-step .formula  { color: #64748b; font-size: 11.5px; }
  .elo-tie { color: #94a3b8; font-family: system-ui, sans-serif; font-size: 13px; }

  /* ── Таблица рейтингов ── */
  .ratings-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }
  .ratings-table th { padding: 6px 10px; text-align: left; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; border-bottom: 2px solid var(--border); }
  .ratings-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
  .ratings-table tr:last-child td { border-bottom: none; }
  .ratings-table tr:hover td { background: #f8fafc; }
  .elo-num { font-weight: 700; font-size: 14px; }
  .rank-1 { color: #f59e0b; }
  .rank-2 { color: #94a3b8; }
  .rank-3 { color: #cd7c2f; }

  /* ── Диаграмма ── */
  .chart-wrap { height: 200px; position: relative; margin-top: 12px; }

  /* ── Вспомогательные ── */
  .msg { text-align: center; color: var(--muted); font-size: 14px; padding: 36px 0; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 3px solid var(--border); border-top-color: var(--blue); border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #1e293b; color: #fff; padding: 9px 18px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .3s; pointer-events: none; z-index: 100; }
  .toast.show { opacity: 1; }
  .auto-hint { font-size: 11px; color: var(--muted); margin-top: 5px; }
</style>
</head>
<body>

<!-- ══ ШАПКА ══════════════════════════════════════════════════ -->
<header>
  <span style="font-size:20px;">⚔️</span>
  <h1>Мини LM Arena</h1>
  <span class="badge" id="models-badge">⏳ определяю модели Ollama…</span>
</header>

<div class="main">

  <!-- ══ БЛОК ВВОДА ВОПРОСА ══════════════════════════════════ -->
  <div class="section">
    <div class="section-title">Вопрос</div>
    <div class="ask-row">
      <textarea id="question" placeholder="Задайте вопрос — или нажмите «Авто» чтобы сгенерировать случайный"></textarea>
      <div class="ask-btns">
        <button class="btn btn-primary" id="btn-ask" onclick="doAsk()">▶ Спросить</button>
        <button class="btn btn-auto"    id="btn-auto" onclick="doAutoQuestion()">🎲 Авто</button>
      </div>
    </div>
    <div class="auto-hint" id="auto-hint"></div>
  </div>

  <!-- ══ АРЕНА: ОТВЕТЫ И ГОЛОСОВАНИЕ ═══════════════════════════ -->
  <div id="arena-area">
    <div class="section">
      <div class="msg">Задайте вопрос выше, чтобы начать сравнение</div>
    </div>
  </div>

  <!-- ══ ELO-ВЫЧИСЛЕНИЯ (показываем после голосования) ══════════ -->
  <div id="elo-calc-wrap" style="display:none;">
    <div class="elo-panel" id="elo-calc-panel">
      <div class="ep-title">📐 Вычисление Elo</div>
      <div id="elo-calc-body"></div>
    </div>
  </div>

  <!-- ══ СТАТИСТИКА И ДИАГРАММА ══════════════════════════════════ -->
  <div class="section" id="stats-box" style="display:none;">
    <div class="section-title">📊 Рейтинг Elo — текущие результаты</div>
    <table class="ratings-table" id="ratings-table">
      <thead><tr>
        <th>#</th><th>Модель</th><th>Elo</th>
        <th>Побед</th><th>Поражений</th><th>Ничьих</th><th>Битв</th><th>Win%</th>
      </tr></thead>
      <tbody id="ratings-tbody"></tbody>
    </table>
    <div class="chart-wrap">
      <canvas id="eloChart"></canvas>
    </div>
  </div>

</div><!-- /main -->

<div class="toast" id="toast"></div>

<!-- ══ JAVASCRIPT ═══════════════════════════════════════════════ -->
<script>
// ── Глобальное состояние ──────────────────────────────────────
let currentLeft = null;   // id модели которая отображается слева
let currentRight = null;  // id модели которая отображается справа
let currentBattleId = null; // id текущей битвы для связывания /ask и /vote в JSON-журнале
let eloChart = null;      // экземпляр Chart.js (нужен для destroy перед пересозданием)
let voteDone = false;     // флаг: голос уже отдан (запрещает повторное голосование)
const COLORS = ['#3b82f6','#22c55e','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899'];

// ── Загрузка списка моделей при старте страницы ───────────────
// Вызываем GET /models — бэкенд возвращает установленные chat-модели Ollama
async function loadModels() {
  const badge = document.getElementById('models-badge');
  try {
    const r = await fetch('/models');
    const d = await r.json();
    if (d.models.length === 0) {
      badge.textContent = '⚠️ Нет моделей Ollama — установите минимум две: ollama pull <model>';
      badge.style.color = '#ef4444';
    } else {
      // Склоняем слово "модель" по-русски
      const n = d.models.length;
      const word = n === 1 ? 'доступна' : (n < 5 ? 'доступны' : 'доступно');
      badge.textContent = `✅ ${n} ${word}: ${d.models.map(m => m.split('/').pop()).join(', ')}`;
      badge.style.color = '#22c55e';
    }
  } catch(e) {
    badge.textContent = '⚠️ Ollama недоступна (порт 11434)';
    badge.style.color = '#ef4444';
  }
}

// ── Кнопка "Авто" — генерируем вопрос через модель ───────────
async function doAutoQuestion() {
  const btn = document.getElementById('btn-auto');
  const hint = document.getElementById('auto-hint');
  btn.disabled = true;
  hint.textContent = '⏳ Генерирую вопрос…';

  try {
    const r = await fetch('/autoquestion', { method: 'POST' });
    const d = await r.json();
    if (d.error) { hint.textContent = '⚠️ ' + d.error; btn.disabled = false; return; }

    // Вставляем сгенерированный вопрос в поле ввода
    document.getElementById('question').value = d.question;
    // Показываем подсказку: какая модель сгенерировала вопрос
    hint.textContent = `✨ Вопрос сгенерирован моделью: ${d.generator.split('/').pop()}`;
    hint.style.color = '#8b5cf6';
  } catch(e) {
    hint.textContent = '⚠️ Ошибка генерации';
  }
  btn.disabled = false;
}

// ── Основной запрос: задаём вопрос двум моделям ──────────────
async function doAsk() {
  const q = document.getElementById('question').value.trim();
  if (!q) { showToast('Введите вопрос'); return; }

  // Блокируем кнопки, сбрасываем состояние
  const btn = document.getElementById('btn-ask');
  btn.disabled = true;
  voteDone = false;

  // Скрываем панель Elo и статистику пока идёт новый вопрос
  document.getElementById('elo-calc-wrap').style.display = 'none';

  document.getElementById('arena-area').innerHTML =
    '<div class="section"><div class="msg"><span class="spinner"></span>Спрашиваю обе модели параллельно…</div></div>';

  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const d = await r.json();
    if (d.error) { showToast('Ошибка: ' + d.error); btn.disabled = false; return; }

    // Запоминаем текущие модели для последующей отправки голоса
    currentLeft  = d.left;
    currentRight = d.right;
    currentBattleId = d.battle_id;
    renderArena(d);  // рисуем карточки с ответами
  } catch(e) {
    showToast('Ошибка соединения с сервером');
  }
  btn.disabled = false;
}

// ── Рендер арены: две карточки + кнопки голосования ─────────
function renderArena(d) {
  document.getElementById('arena-area').innerHTML = `
    <div class="section">
      <div class="section-title">Ответы — выберите лучший (имена скрыты)</div>
      <div class="arena">
        <div class="card" id="card-left">
          <div class="card-label">⬛ Модель A</div>
          <div class="card-answer">${escHtml(d.ans_left)}</div>
          <div class="card-model" id="name-left">🤖 ${escHtml(d.left)}</div>
        </div>
        <div class="card" id="card-right">
          <div class="card-label">⬛ Модель B</div>
          <div class="card-answer">${escHtml(d.ans_right)}</div>
          <div class="card-model" id="name-right">🤖 ${escHtml(d.right)}</div>
        </div>
      </div>
      <div class="vote-row" style="margin-top:14px;">
        <button class="btn-vote left"  onclick="doVote('left')" >👈 A лучше</button>
        <button class="btn-vote tie"   onclick="doVote('tie')"  >🤝 Ничья</button>
        <button class="btn-vote right" onclick="doVote('right')">B лучше 👉</button>
      </div>
    </div>
  `;
}

// ── Голосование: фиксируем выбор, показываем Elo-вычисление ─
async function doVote(choice) {
  if (voteDone) return;   // защита от двойного клика
  voteDone = true;
  document.querySelectorAll('.btn-vote').forEach(b => b.disabled = true);

  // Раскрываем имена моделей (слепой тест завершён)
  document.getElementById('name-left').style.display  = 'block';
  document.getElementById('name-right').style.display = 'block';

  // Подсвечиваем победителя зелёным, проигравшего делаем тусклее
  if (choice === 'left') {
    document.getElementById('card-left').classList.add('winner');
    document.getElementById('card-right').classList.add('loser');
  } else if (choice === 'right') {
    document.getElementById('card-right').classList.add('winner');
    document.getElementById('card-left').classList.add('loser');
  }

  try {
    const r = await fetch('/vote', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        winner: choice,
        left: currentLeft,
        right: currentRight,
        battle_id: currentBattleId
      })
    });
    const d = await r.json();

    // Показываем пошаговое Elo-вычисление
    renderEloCalc(choice, d.elo_details);

    // Обновляем таблицу и диаграмму
    renderStats(d.ratings);

    showToast(choice === 'tie' ? '🤝 Ничья зафиксирована' : '✅ Голос засчитан!');
  } catch(e) {
    showToast('Ошибка при отправке голоса');
  }
}

// ── Панель пошаговых Elo-вычислений ─────────────────────────
// elo_details содержит: winner, loser, ew_before, el_before,
//                        expected_pct, delta, ew_after, el_after, K
function renderEloCalc(choice, det) {
  const wrap  = document.getElementById('elo-calc-wrap');
  const body  = document.getElementById('elo-calc-body');
  wrap.style.display = 'block';

  if (choice === 'tie' || !det) {
    // При ничьей Elo не меняется — объясняем почему
    body.innerHTML = '<div class="elo-tie">🤝 Ничья — рейтинги не изменились.<br>'
      + 'В упрощённой системе ничья не передаёт очки. '
      + 'В полной системе Elo ничья приравнивается к результату 0.5 для обоих участников.</div>';
    return;
  }

  // Короткие имена для читаемости
  const wName = det.winner.split('/').pop();
  const lName = det.loser.split('/').pop();

  // Строим "учебник" по формуле прямо в UI
  body.innerHTML = `
<div class="elo-step">
  <span class="lbl">Победитель:</span>
  <span class="val up">🏆 ${escHtml(wName)}</span>
  <span class="formula">(Elo до битвы: ${det.ew_before})</span>
</div>
<div class="elo-step">
  <span class="lbl">Проигравший:</span>
  <span class="val down">💀 ${escHtml(lName)}</span>
  <span class="formula">(Elo до битвы: ${det.el_before})</span>
</div>
<div class="elo-step" style="margin-top:8px;">
  <span class="lbl">Шаг 1 — Ожидаемая вероятность:</span>
  <span class="val">${det.expected_pct}%</span>
  <span class="formula">= 1 / (1 + 10^((${det.el_before} − ${det.ew_before}) / 400))</span>
</div>
<div class="elo-step">
  <span class="lbl">Шаг 2 — Δ (изменение рейтинга):</span>
  <span class="val up">+${det.delta}</span>
  <span class="formula">= K × (1 − expected) = ${det.K} × (1 − ${(det.expected_pct/100).toFixed(3)}) = ${det.delta}</span>
</div>
<div class="elo-step" style="margin-top:8px;">
  <span class="lbl">Новый Elo победителя:</span>
  <span class="val up">${det.ew_after}</span>
  <span class="formula">= ${det.ew_before} + ${det.delta} <span style="color:#4ade80">▲+${det.delta}</span></span>
</div>
<div class="elo-step">
  <span class="lbl">Новый Elo проигравшего:</span>
  <span class="val down">${det.el_after}</span>
  <span class="formula">= ${det.el_before} − ${det.delta} <span style="color:#f87171">▼−${det.delta}</span></span>
</div>
<div class="elo-step" style="margin-top:8px; color:#475569; font-size:11.5px;">
  <span>Сумма Elo сохранилась: ${det.ew_before} + ${det.el_before} = ${det.ew_after} + ${det.el_after} = ${det.ew_before + det.el_before}</span>
</div>`;
}

// ── Таблица рейтингов + Chart.js диаграмма ──────────────────
function renderStats(ratings) {
  document.getElementById('stats-box').style.display = 'block';

  // Сортируем по Elo по убыванию
  const sorted = Object.entries(ratings).sort((a, b) => b[1].elo - a[1].elo);

  // ── Таблица ──
  const tbody = document.getElementById('ratings-tbody');
  tbody.innerHTML = '';
  sorted.forEach(([id, v], i) => {
    const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : '';
    const shortName = id.split('/').pop();
    const winPct = v.battles > 0 ? ((v.wins / v.battles) * 100).toFixed(0) + '%' : '—';
    const eloDelta = v.elo - 1000;   // отклонение от стартового рейтинга
    const deltaStr = eloDelta >= 0 ? `+${eloDelta.toFixed(1)}` : eloDelta.toFixed(1);
    const deltaColor = eloDelta > 0 ? '#22c55e' : eloDelta < 0 ? '#ef4444' : '#94a3b8';
    tbody.innerHTML += `<tr>
      <td class="${rankClass}" style="font-weight:700;">${i===0?'🥇':i===1?'🥈':i===2?'🥉':i+1}</td>
      <td title="${escHtml(id)}">${escHtml(shortName)}</td>
      <td><span class="elo-num ${rankClass}">${v.elo}</span>
          <span style="font-size:11px;color:${deltaColor};margin-left:4px;">(${deltaStr})</span></td>
      <td style="color:#22c55e;font-weight:600;">${v.wins}</td>
      <td style="color:#ef4444;">${v.losses}</td>
      <td style="color:#94a3b8;">${v.ties}</td>
      <td>${v.battles}</td>
      <td style="font-weight:600;">${winPct}</td>
    </tr>`;
  });

  // ── Chart.js столбчатая диаграмма ──
  const labels = sorted.map(([id]) => id.split('/').pop().substring(0, 20));
  const elos   = sorted.map(([, v]) => v.elo);
  const bgColors = elos.map((_, i) => COLORS[i % COLORS.length] + 'bb');
  const bdColors = elos.map((_, i) => COLORS[i % COLORS.length]);

  const ctx = document.getElementById('eloChart').getContext('2d');
  if (eloChart) eloChart.destroy();   // важно: уничтожить старый экземпляр

  eloChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Elo рейтинг',
        data: elos,
        backgroundColor: bgColors,
        borderColor: bdColors,
        borderWidth: 2,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            // Подсказка при наведении — показываем полную статистику
            afterLabel: (ctx) => {
              const [, v] = sorted[ctx.dataIndex];
              const pct = v.battles > 0 ? ((v.wins/v.battles)*100).toFixed(0) : 0;
              return `Победы: ${v.wins}  Поражения: ${v.losses}  Ничьи: ${v.ties}  Win%: ${pct}%`;
            }
          }
        }
      },
      scales: {
        // beginAtZero: false — иначе столбики почти одинаковые при Elo 980 и 1020
        y: { beginAtZero: false, min: Math.min(...elos) - 40, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
}

// ── Утилиты ──────────────────────────────────────────────────

// Экранирует HTML-символы чтобы ответы моделей не сломали разметку
function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Всплывающее уведомление внизу экрана
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

// ── Запуск при загрузке страницы ─────────────────────────────
// Сразу определяем какие модели Ollama доступны
loadModels();
</script>
</body>
</html>"""


DEMO_QUESTIONS = [
    "Объясни разницу между BLEU и ROUGE на примере перевода и суммаризации.",
    "Что такое перплексия и почему она не гарантирует полезность ответа LLM?",
    "Напиши функцию Python для расчёта факториала с обработкой неверного ввода.",
    "Почему в Chatbot Arena важно скрывать имена моделей до голосования?",
    "Объясни закон Гудхарта применительно к ML-бенчмаркам.",
    "Сравни монолитную и микросервисную архитектуру в трёх пунктах.",
    "Что такое BERTScore и чем он отличается от подсчёта n-грамм?",
    "Напиши SQL-запрос для выбора топ-5 пользователей по сумме заказов.",
    "Как position bias и length bias искажают человеческую оценку ответов?",
    "Когда лучше использовать LLM-as-a-Judge, а когда human evaluation?",
]


def demo_answer(model_id, question):
    """Детерминированный ответ для автономного демо-экспорта без Ollama."""
    if model_id.endswith("concise"):
        return (
            f"{model_id}: краткий ответ на вопрос «{question}». "
            "Сначала нужно определить критерий качества, затем сравнить ответы "
            "по этому критерию и зафиксировать результат."
        )
    return (
        f"{model_id}: развёрнутый ответ на вопрос «{question}». "
        "Важно учитывать задачу, формат оценки, возможные смещения пользователя "
        "и воспроизводимость результата. Для открытых ответов удобно использовать "
        "попарное сравнение и Elo, потому что человеку проще выбрать лучший ответ, "
        "чем ставить абсолютную оценку."
    )


def run_demo_export(path=None):
    """
    Создаёт пример JSON минимум на 10 вопросов.
    Нужен для проверки формата экспорта, когда Ollama недоступна.
    """
    ratings.clear()
    test_history.clear()
    pending_battles.clear()

    models = ["demo/concise", "demo/detailed"]
    votes = ["right", "right", "left", "tie", "right", "left", "right", "tie", "left", "right"]

    for question, winner in zip(DEMO_QUESTIONS, votes):
        left, right = models
        ans_left = demo_answer(left, question)
        ans_right = demo_answer(right, question)
        before, after, elo_details = apply_vote(left, right, winner)
        record_test(
            question=question,
            left=left,
            right=right,
            ans_left=ans_left,
            ans_right=ans_right,
            winner=winner,
            before=before,
            after=after,
            elo_details=elo_details,
            battle_id=str(uuid4()),
        )

    payload = save_test_history(path or EXPORT_PATH)
    print(f"Демо-экспорт сохранён: {path.resolve()}")
    print(f"Тестов в JSON: {payload['total_tests']}")


# ********************* ТОЧКА ВХОДА *********************

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Мини LM Arena")
    parser.add_argument(
        "--demo-export",
        action="store_true",
        help="создать arena_tests.json с 10 демо-тестами без запуска Ollama"
    )
    parser.add_argument(
        "--export-path",
        default=str(EXPORT_PATH),
        help="путь к JSON-файлу экспорта"
    )
    args = parser.parse_args()

    if args.demo_export:
        run_demo_export(Path(args.export_path))
        raise SystemExit(0)

    print(f"\n  Мини LM Arena: http://localhost:{PORT}")
    print(f"\n  Требования:")
    print(f"    1. Ollama запущена локально (порт 11434)")
    print(f"    2. Минимум 2 chat-модели установлены: ollama pull <model>")
    print(f"    3. pip install flask")
    print(f"\n  Ollama API: {OLLAMA_CHAT_API}")
    print(f"  Настройки: K={ELO_K}, start_elo={ELO_START}\n")
    app.run(debug=True, port=PORT)
