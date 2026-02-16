# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chef Claude — a realtime voice AI cooking assistant powered by Claude and LiveKit. Users talk hands-free while cooking and get expert guidance via voice, with optional camera vision, on-screen recipe cards, timers, and step tracking.

## Architecture

### Backend — `agent.py`

Python LiveKit voice agent with a multi-agent pipeline:

- **Pipeline**: Deepgram STT → Claude LLM (Opus) → ElevenLabs/Cartesia TTS, with Silero VAD
- **Three agent classes** with seamless handoffs via `ctx.session.update_agent()`:
  - `OnboardingAgent` — gets to know the user (name, background, comfort level, goals), saves profile to `user_profiles/`
  - `ChefAgent` — main cooking assistant, suggests dishes, starts recipes, manages camera requests
  - `RecipeAgent` — step-by-step recipe guidance with step tracking, timers, substitutions
- **Vision**: Custom frame capture from `rtc.VideoStream` at ~2 FPS into a ring buffer (3 frames). Frames are encoded to 1024x1024 JPEG and injected into `llm.ChatContext` via `on_user_turn_completed`
- **Context management**: Image stripping from old messages, async summarization via Haiku when chat items exceed 24, hard trim at 40
- **Data channel** (`_publish_data`): Agent→frontend communication for timers, camera requests, recipe cards, dish suggestions, step updates, recipe refreshes
- **Frontend→Agent data**: Dish selection taps are received via room `data_received` event, interrupts current speech, and triggers `generate_reply`
- **Web search**: Uses Anthropic API `web_search_20250305` tool via Haiku to find recipe tutorials, then fetches OG metadata for link previews
- **TTS switching**: `TTS_PROVIDER` env var (`cartesia` or `elevenlabs`), `TTS_VOICE_ID` for voice selection

### Frontend — `frontend/`

Next.js (App Router, TypeScript, Tailwind) web UI:

- **`app/page.tsx`** — Main UI with all voice, video, and recipe components
- **`app/api/token/route.ts`** — LiveKit token generation
- **Key components**:
  - `VoiceAssistantUI` — orchestrates all UI state, listens for data channel messages on topics: `timer`, `camera_request`, `recipe`, `suggestions`
  - `RecipeCard` — collapsible recipe card (preview → expanded) with ingredients/steps tabs, servings, prep time, step highlighting, tutorial link preview with OG image
  - `DishSuggestions` — clickable dish option cards, publishes selection back via data channel
  - `TimerDisplay` — countdown timers with Web Audio API chime alerts and browser notifications
- **Layout**: Fixed bottom controls (mic/camera/disconnect + visualizer), scrollable content area above for recipe cards, suggestions, timers, camera preview

### Data Channel Topics

| Topic | Direction | Messages |
|-------|-----------|----------|
| `timer` | Agent→Frontend | `set_timer` |
| `camera_request` | Agent→Frontend | `request_camera` |
| `recipe` | Agent→Frontend | `recipe_start`, `recipe_update` (tutorial link + OG), `recipe_refresh` (substitutions), `step_update`, `recipe_end` |
| `suggestions` | Agent→Frontend | `dish_suggestions` |
| `dish_selection` | Frontend→Agent | `select_dish` |

### Agent Tools

**ChefAgent**: `request_camera`, `set_timer`, `suggest_dishes`, `start_recipe`
**RecipeAgent**: `update_step`, `update_recipe`, `request_camera`, `set_timer`, `finish_recipe`
**OnboardingAgent**: `save_profile_and_start_cooking`

## Commands

### Backend (from project root)
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run agent.py dev          # dev mode with hot reload
uv run agent.py console      # local test without LiveKit Cloud
uv run agent.py start        # production mode
```

### Frontend (from frontend/)
```bash
nvm use 20
npm run dev                   # dev server at localhost:3000
npm run build                 # production build
```

## Environment Variables

### Backend (`.env.local`)
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — LiveKit credentials
- `ANTHROPIC_API_KEY` — Claude API (used for LLM, web search, and async summarization)
- `DEEPGRAM_API_KEY` — Speech-to-text
- `CARTESIA_API_KEY` — TTS (if using Cartesia)
- `ELEVEN_API_KEY` — TTS (if using ElevenLabs)
- `TTS_PROVIDER` — `cartesia` or `elevenlabs` (default: cartesia)
- `TTS_VOICE_ID` — Optional voice ID for your TTS provider

### Frontend (`frontend/.env.local`)
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`

## Key Dependencies

- Python ≥3.12, managed by `uv`
- Node ≥20 (via `nvm use 20`)
- `livekit-agents[anthropic,silero,turn-detector,images]~=1.3`
- `livekit-plugins-deepgram~=1.3`, `livekit-plugins-cartesia~=1.3`, `livekit-plugins-elevenlabs~=1.3`
- `anthropic` (SDK, also used for web search and Haiku summarization)
- `httpx` (transitive via anthropic, used for OG metadata fetching)
- `@livekit/components-react`, `livekit-client`, `livekit-server-sdk`

## Known Transient Errors

- **Deepgram `1011` timeout**: WebSocket idle timeout on free tier. Framework auto-reconnects (`retryable=True`). Harmless.
- **TTS `no audio frames`**: Intermittent ElevenLabs/Cartesia failure. Framework retries up to 3x. Users may notice a brief pause.
