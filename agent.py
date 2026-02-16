import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

import anthropic as anthropic_sdk
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    RunContext,
    cli,
    function_tool,
    llm,
    room_io,
)
from livekit import rtc
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
from livekit.plugins import anthropic, cartesia, deepgram, elevenlabs, silero

logger = logging.getLogger("chef_claude")

# Stored reference to the room so tools can publish data messages
_current_room: rtc.Room | None = None

# Ring buffer of recent video frames (captures ~2 FPS, keeps last 3)
_frame_buffer: deque[rtc.VideoFrame] = deque(maxlen=3)

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local").lower()

# --- Storage layer ---
# Set STORAGE_BACKEND=redis and UPSTASH_REDIS_URL for cloud deploys.
# Defaults to local JSON files for dev.

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis as _redis
        url = os.environ.get("UPSTASH_REDIS_URL", "")
        if not url:
            raise RuntimeError("UPSTASH_REDIS_URL env var required when STORAGE_BACKEND=redis")
        _redis_client = _redis.from_url(url, decode_responses=True)
    return _redis_client

if STORAGE_BACKEND == "local":
    PROFILES_DIR = Path("user_profiles")
    PROFILES_DIR.mkdir(exist_ok=True)
    GROCERY_DIR = Path("grocery_lists")
    GROCERY_DIR.mkdir(exist_ok=True)

# Context management settings
MAX_CHAT_ITEMS = 40
KEEP_IMAGES_IN_LAST_N = 2
# Trigger async summarization when item count exceeds this (before hitting MAX)
SUMMARIZE_THRESHOLD = 24
# Keep this many recent items untouched when summarizing
KEEP_RECENT_ITEMS = 8

# Async summarization state
_summarization_lock = asyncio.Lock()
_summarization_in_progress = False

VOICE_STYLE = """\
## Important: You are a voice-only assistant
Everything you say is spoken aloud via text-to-speech. Never include stage directions, \
actions, or emotive descriptions like *smiles*, *rubs hands*, *laughs*, (pauses), etc. \
Express warmth and personality purely through your word choice and tone.

## Camera / Vision
The camera starts OFF by default to respect the user's privacy. If the user says something \
that would benefit from visual context (e.g. "does this look done?", "what can I make with \
these?", "how does this look?", "check this out"), call the `request_camera` tool to prompt \
them to enable their camera. Once enabled, you can see their video feed and use it to:
- Identify ingredients they show you
- Assess cooking progress (color, texture, doneness)
- Spot potential issues (too much heat, uneven chopping, etc.)
- Give feedback on plating and presentation
Never assume the camera is on. If you don't see any images in the conversation, the camera \
is off — call `request_camera` before commenting on anything visual.
When images are present, you receive up to 3 frames captured over ~1.5 seconds, ordered \
oldest to newest. The last image is the most current. Use the sequence to understand motion \
or changes (e.g. stirring, flipping, pouring)."""

ONBOARDING_PROMPT = f"""\
You are Chef Claude, a world-class culinary assistant with the expertise of a \
Michelin-starred chef. You are meeting a new user for the first time.

{VOICE_STYLE}

## Your task: Get to know this person

Walk them through a warm, natural get-to-know-you conversation. \
Don't rush — treat it like meeting someone at a dinner party. One question at a time, and \
react genuinely to their answers before moving on.

1. **Greet and get their name.** Start with a warm welcome, introduce yourself as Chef Claude, \
and ask for their full name. Use their first name naturally from then on.

2. **Learn their culinary background.** Ask about the kind of food they grew up eating or \
love to cook. Keep it regional and cultural (e.g. "Southern comfort food", "home-style Indian", \
"Mediterranean") — don't probe for specific countries unless they volunteer it. React with \
genuine interest and share a brief, relevant comment to build rapport.

3. **Dietary-Restrictions or Preferences** Ask if the user has any dietary restrictions or preferences \
that should be kept in mind when suggesting and crafting recipes. If any let them know you understood by 
confirming with a brief remark like 'Noted'. 

3. **Gauge their cooking comfort level.** Ask how comfortable they are in the kitchen. \
Are they a total beginner, someone who can follow a recipe, or do they like to improvise? \
Frame it casually — no one should feel judged.

4. **Understand their goals.** Ask what they're hoping to get out of cooking with you. \
Some examples to weave in naturally: picking up quick weeknight meals, mastering a specific \
cuisine, impressing guests, becoming a confident home chef, or just having fun experimenting.

Once you have learned all four things, call the `save_profile_and_start_cooking` tool with a \
compact JSON summary. Then say a brief transition like "Alright, let's get cooking!" — do NOT \
repeat the full summary back to them, just show you remember by using their name."""

CHEF_PROMPT = """\
You are Chef Claude, a world-class culinary assistant with the expertise of a \
Michelin-starred chef. You guide home cooks through recipes with clear, concise, step-by-step \
instructions optimized for voice.

{voice_style}

Your style:
- Keep responses short and conversational — the user is cooking hands-free
- Give one step at a time unless asked for an overview
- Proactively warn about timing, temperature, and food safety
- Suggest substitutions when ingredients are missing
- Adapt to the cook's skill level based on their questions
- Be encouraging and enthusiastic about cooking

## About this user
{user_profile}

Use what you know about them to personalize suggestions — their background, comfort level, \
and goals. Address them by their first name. Don't re-introduce yourself or repeat their profile \
back to them — just cook together like old friends.

When the user decides on a recipe, ALWAYS ask how many people they're cooking for before \
calling `start_recipe`. A quick "How many are we feeding tonight?" is enough. Then scale the \
ingredient quantities and the recipe accordingly. Pass the servings count to `start_recipe`. \
You can generate a recipe yourself based on what they want, or if they ask for something very \
specific, do your best to create an authentic version. Tailor the recipe complexity to their \
comfort level.

When suggesting multiple dish options, call `suggest_dishes` to show the options on the user's \
screen as clickable cards. The user can tap one or just say their choice out loud — both work. \
Keep suggestions to 2-4 options with a short description for each.

## Dish history
If the user's profile includes recent dishes, use that to avoid repeating the same meals and to \
suggest complementary or new recipes. Reference their past cooking naturally, e.g. "Last time \
you made pasta, want to try something different tonight?"

## Grocery list
When the user decides on a recipe or mentions needing to shop, offer to save the ingredients to \
their grocery list using `save_to_grocery_list`. The list persists across sessions so they can \
reference it later at the store. If they ask to clear or reset it, use `clear_grocery_list`. \
The user can also ask you questions about ingredients while shopping (e.g. "is this the right \
kind of flour?") — use your expertise and the camera if available to help. When the user says \
they're at the store or wants to see their list, call `show_grocery_list` to pull it up. If \
they show you a product via camera, help them determine if it's the right item for their recipe."""

RECIPE_PROMPT = """\
You are Chef Claude, guiding {first_name} through a recipe step by step.

{voice_style}

Your style:
- Give ONE step at a time, then wait for the user to say they're ready
- Be specific about quantities, temperatures, and timing
- If the user asks "what's next?" or "done", move to the next step
- If they ask to repeat, repeat the current step clearly
- If they ask to go back, return to the previous step
- Proactively mention timing cues ("you'll know it's ready when...")
- If something goes wrong, stay calm and help them recover
- Offer encouragement as they progress
- When a step involves waiting (boiling, baking, resting, etc.), proactively call `set_timer` \
so the user gets an alert — don't just tell them the time, actually set it. After setting a \
timer, briefly confirm it verbally (e.g. "I've set a 10 minute timer for the pasta")

## Step tracking
IMPORTANT: Every time you start explaining a step, call `update_step` with the 1-based step \
number BEFORE speaking about it. This updates the user's screen to highlight the current step. \
Call it when moving forward, backward, or repeating. Start by calling `update_step(step_number=1)` \
for the first step.

## Substitutions and adjustments
When the user asks to substitute an ingredient, adjust quantities, modify a step, or make any \
change to the recipe, call `update_recipe` with the FULL updated ingredients list and steps. \
This refreshes the recipe card on their screen while keeping their current step highlighted. \
Always pass the complete lists (not just the changed parts).

## The Recipe
{recipe}

## Current cook's profile
{user_profile}

When the user finishes the recipe or wants to stop cooking, call `finish_recipe` to return \
to the main conversation.

## Grocery list
If the user asks to save ingredients or see their grocery list, use `save_to_grocery_list` \
or `show_grocery_list`. If they mention being at the store, show the list and help them \
identify the right products using the camera if available."""


async def _search_recipe_link(recipe_title: str) -> dict | None:
    """Use Anthropic web search to find a cooking video or tutorial for the recipe.

    Returns {"url": ..., "title": ..., "source": ...} or None.
    """
    import re

    def _extract_link(url: str, title: str) -> dict:
        url = url.rstrip(".,;)\"'")
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        source = domain_match.group(1) if domain_match else "web"
        return {"url": url, "title": title.strip(), "source": source}

    try:
        client = anthropic_sdk.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{
                "role": "user",
                "content": (
                    f"Find the single best video or recipe tutorial for making: {recipe_title}\n\n"
                    "Prefer high-quality sources like YouTube cooking channels, NYT Cooking, "
                    "Bon Appetit, Serious Eats, Food Network, Epicurious, or popular food blogs. "
                    "Pick whichever result is most helpful — a video is ideal but a great written "
                    "tutorial with photos works too.\n\n"
                    "Return EXACTLY this format, nothing else:\n"
                    "URL: <full url>\n"
                    "TITLE: <title of the page/video>\n"
                ),
            }],
        )

        # Strategy 1: Parse URL:/TITLE: from text blocks
        for block in response.content:
            if hasattr(block, "text"):
                url_match = re.search(r'URL:\s*(https?://\S+)', block.text)
                title_match = re.search(r'TITLE:\s*(.+)', block.text)
                if url_match:
                    title = title_match.group(1) if title_match else recipe_title
                    return _extract_link(url_match.group(1), title)

        # Strategy 2: Extract any URL from text blocks
        for block in response.content:
            if hasattr(block, "text"):
                urls = re.findall(r'https?://\S+', block.text)
                if urls:
                    return _extract_link(urls[0], recipe_title)

        # Strategy 3: Pull URL from web search result citations
        for block in response.content:
            if hasattr(block, "text") and hasattr(block, "citations") and block.citations:
                for cite in block.citations:
                    if hasattr(cite, "url") and cite.url:
                        title = getattr(cite, "title", recipe_title) or recipe_title
                        return _extract_link(cite.url, title)

        # Strategy 4: Pull from web_search_tool_result blocks directly
        for block in response.content:
            if getattr(block, "type", None) == "web_search_tool_result":
                results = getattr(block, "content", [])
                if results and hasattr(results[0], "url"):
                    title = getattr(results[0], "title", recipe_title) or recipe_title
                    return _extract_link(results[0].url, title)

    except Exception:
        logger.exception("Recipe link search failed for: %s", recipe_title)
    return None


async def _fetch_link_preview(url: str) -> dict | None:
    """Fetch Open Graph metadata from a URL for a rich link preview.

    Returns {"og_image": ..., "og_title": ..., "og_description": ...} or None.
    """
    import re
    import httpx

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ChefClaude/1.0)",
            })
            if resp.status_code != 200:
                return None
            html = resp.text[:50_000]  # only parse the head

        def _og(prop: str) -> str | None:
            # Match both property= and name= variants
            m = re.search(
                rf'<meta[^>]+(?:property|name)=["\']og:{prop}["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if not m:
                # Try reversed attribute order (content before property)
                m = re.search(
                    rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:{prop}["\']',
                    html, re.IGNORECASE,
                )
            return m.group(1) if m else None

        og_image = _og("image")
        og_title = _og("title")
        og_description = _og("description")

        if not og_image:
            # Fallback: try twitter:image
            m = re.search(
                r'<meta[^>]+(?:property|name)=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if not m:
                m = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']twitter:image["\']',
                    html, re.IGNORECASE,
                )
            og_image = m.group(1) if m else None

        if og_image or og_title:
            return {
                "og_image": og_image,
                "og_title": og_title,
                "og_description": og_description,
            }
    except Exception:
        logger.debug("Link preview fetch failed for: %s", url)
    return None


async def _publish_data(topic: str, data: dict) -> None:
    """Send a data message to the frontend via LiveKit data channel."""
    if _current_room and _current_room.local_participant:
        msg = json.dumps(data)
        await _current_room.local_participant.publish_data(msg, topic=topic)


async def _publish_timer(label: str, duration_seconds: int) -> None:
    """Send a timer event to the frontend via LiveKit data channel."""
    if _current_room and _current_room.local_participant:
        msg = json.dumps({
            "type": "set_timer",
            "label": label,
            "duration_seconds": duration_seconds,
        })
        await _current_room.local_participant.publish_data(
            msg, topic="timer"
        )


def _format_profile(profile: dict) -> str:
    lines = [
        f"Name: {profile['first_name']} ({profile['full_name']})",
        f"Culinary background: {profile['culinary_background']}",
        f"Comfort level: {profile['comfort_level']}",
        f"Goals: {profile['goals']}",
    ]
    history = profile.get("dish_history", [])
    if history:
        recent = history[-5:]  # last 5 dishes
        dishes = ", ".join(f"{d['title']} ({d['date']})" for d in recent)
        lines.append(f"Recent dishes cooked: {dishes}")
    return "\n".join(lines)


def load_profile(user_id: str) -> dict | None:
    if STORAGE_BACKEND == "redis":
        data = _get_redis().get(f"profile:{user_id}")
        return json.loads(data) if data else None
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_profile(user_id: str, profile: dict) -> None:
    if STORAGE_BACKEND == "redis":
        _get_redis().set(f"profile:{user_id}", json.dumps(profile))
        return
    path = PROFILES_DIR / f"{user_id}.json"
    path.write_text(json.dumps(profile, indent=2))


def load_grocery_list(user_id: str) -> list[dict]:
    if STORAGE_BACKEND == "redis":
        data = _get_redis().get(f"grocery:{user_id}")
        return json.loads(data) if data else []
    path = GROCERY_DIR / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_grocery_list(user_id: str, items: list[dict]) -> None:
    if STORAGE_BACKEND == "redis":
        _get_redis().set(f"grocery:{user_id}", json.dumps(items))
        return
    path = GROCERY_DIR / f"{user_id}.json"
    path.write_text(json.dumps(items, indent=2))


class OnboardingAgent(Agent):
    def __init__(self, user_id: str, user_profile: dict) -> None:
        super().__init__(instructions=ONBOARDING_PROMPT)
        self._user_id = user_id
        self._user_profile = user_profile

    @function_tool(
        description="Save the user's profile after onboarding and transition to cooking. "
        "Call this once you've learned their name, culinary background, comfort level, and goals."
    )
    async def save_profile_and_start_cooking(
        self,
        ctx: RunContext,
        full_name: str,
        first_name: str,
        culinary_background: str,
        comfort_level: str,
        goals: str,
    ) -> str:
        profile = {
            "full_name": full_name,
            "first_name": first_name,
            "culinary_background": culinary_background,
            "comfort_level": comfort_level,
            "goals": goals,
        }
        save_profile(self._user_id, profile)
        self._user_profile.update(profile)

        chef = ChefAgent(self._user_profile, user_id=self._user_id)
        ctx.session.update_agent(chef)
        return "Profile saved. Now in cooking mode."


class ChefAgent(Agent):
    def __init__(self, user_profile: dict, user_id: str = "default_user") -> None:
        super().__init__(
            instructions=CHEF_PROMPT.format(
                voice_style=VOICE_STYLE,
                user_profile=_format_profile(user_profile),
            ),
        )
        self._user_profile = user_profile
        self._user_id = user_id

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Strip old images, inject new video frames, and trigger summarization if needed."""
        _strip_old_images(turn_ctx)
        _inject_video_frames(turn_ctx, new_message)
        # Trigger background summarization on the agent's persistent chat context
        # (turn_ctx is a copy, so we check the real one)
        await _maybe_summarize(self.chat_ctx)

    @function_tool(
        description="Request the user to enable their camera. Call this when the user says "
        "something that would benefit from visual context, like 'does this look done?', "
        "'what can I make with these?', 'look at this', etc."
    )
    async def request_camera(self, ctx: RunContext) -> str:
        await _publish_data("camera_request", {"type": "request_camera"})
        return "Camera request sent to the user's screen. Wait for them to enable it before commenting on visuals."

    @function_tool(
        description="Set a cooking timer. Use this whenever the user needs to wait "
        "(e.g. simmering, baking, resting). The timer will alert the user when done."
    )
    async def set_timer(
        self,
        ctx: RunContext,
        label: str,
        duration_minutes: float,
    ) -> str:
        seconds = int(duration_minutes * 60)
        await _publish_timer(label, seconds)
        return f"Timer set: {label} for {duration_minutes} minutes."

    @function_tool(
        description="Show dish options on the user's screen as clickable cards. Call this when "
        "you're suggesting several dishes for the user to choose from. Each option needs a title "
        "and a short one-line description. The user can tap to select or just say their choice."
    )
    async def suggest_dishes(
        self,
        ctx: RunContext,
        options: str,
    ) -> str:
        """options should be a JSON array of objects with 'title' and 'description' keys."""
        import json as _json
        try:
            parsed = _json.loads(options)
        except Exception:
            return "Failed to parse options. Pass a valid JSON array."
        await _publish_data("suggestions", {
            "type": "dish_suggestions",
            "options": parsed,
        })
        return (
            "Dish options are now displayed on the user's screen. "
            "Wait for them to pick one — they can tap or speak their choice."
        )

    @function_tool(
        description="Show the user's grocery list on their screen. Call this when the user asks "
        "to see their grocery list, check what they need to buy, or when they're heading to the store."
    )
    async def show_grocery_list(self, ctx: RunContext) -> str:
        items = load_grocery_list(self._user_id)
        await _publish_data("grocery_list", {
            "type": "grocery_list_show",
            "items": items,
        })
        if not items:
            return "The grocery list is empty — nothing saved yet."
        total = sum(len(g["ingredients"]) for g in items)
        return f"Grocery list is now on screen with {total} items across {len(items)} recipe(s)."

    @function_tool(
        description="Save ingredients to the user's grocery list. Call this when the user wants "
        "to save a recipe's ingredients for later shopping, or when they mention specific items "
        "they need to buy. Pass the recipe title and a JSON array of ingredient strings."
    )
    async def save_to_grocery_list(
        self,
        ctx: RunContext,
        recipe_title: str,
        ingredients: str,
    ) -> str:
        """ingredients should be a JSON array of ingredient strings."""
        import json as _json
        try:
            parsed = _json.loads(ingredients)
        except Exception:
            return "Failed to parse ingredients. Pass a valid JSON array of strings."
        items = load_grocery_list(self._user_id)
        items.append({"recipe": recipe_title, "ingredients": parsed, "checked": []})
        save_grocery_list(self._user_id, items)
        # Publish to frontend
        await _publish_data("grocery_list", {
            "type": "grocery_list_update",
            "items": items,
        })
        return f"Saved {len(parsed)} ingredients from {recipe_title} to your grocery list."

    @function_tool(
        description="Clear the user's grocery list entirely, or remove ingredients for a specific "
        "recipe. Call this when the user says they've finished shopping or wants to start fresh."
    )
    async def clear_grocery_list(
        self,
        ctx: RunContext,
        recipe_title: str = "",
    ) -> str:
        if recipe_title:
            items = load_grocery_list(self._user_id)
            items = [i for i in items if i.get("recipe") != recipe_title]
            save_grocery_list(self._user_id, items)
        else:
            items = []
            save_grocery_list(self._user_id, items)
        await _publish_data("grocery_list", {
            "type": "grocery_list_update",
            "items": items,
        })
        return "Grocery list cleared." if not recipe_title else f"Removed {recipe_title} from grocery list."

    @function_tool(
        description="Start guiding the user through a recipe. Call this when the user has "
        "decided what to cook, confirmed the number of servings, and you've generated or "
        "found a suitable recipe scaled to that serving size. Pass the complete recipe as a "
        "formatted string with ingredients list and numbered steps."
    )
    async def start_recipe(
        self,
        ctx: RunContext,
        recipe_title: str,
        servings: int,
        prep_time_minutes: int,
        ingredients: str,
        steps: str,
    ) -> str:
        full_recipe = f"# {recipe_title} (Serves {servings})\n\n## Ingredients\n{ingredients}\n\n## Steps\n{steps}"

        # Parse and publish recipe to frontend asynchronously so we don't block the tool return
        async def _publish_recipe():
            import re
            step_list = [s.strip() for s in re.split(r'\n\d+[\.\)]\s*', steps) if s.strip()]
            ingredient_list = [i.strip() for i in ingredients.strip().split('\n') if i.strip()]
            await _publish_data("recipe", {
                "type": "recipe_start",
                "title": recipe_title,
                "servings": servings,
                "prep_time_minutes": prep_time_minutes,
                "ingredients": ingredient_list,
                "steps": step_list,
            })

            # Search for a video/tutorial and fetch its link preview
            link = await _search_recipe_link(recipe_title)
            if link:
                update: dict = {
                    "type": "recipe_update",
                    "tutorial_url": link["url"],
                    "tutorial_title": link["title"],
                    "tutorial_source": link["source"],
                }
                # Fetch OG metadata for a rich preview
                preview = await _fetch_link_preview(link["url"])
                if preview:
                    update["og_image"] = preview.get("og_image")
                    update["og_title"] = preview.get("og_title")
                    update["og_description"] = preview.get("og_description")
                await _publish_data("recipe", update)

        asyncio.create_task(_publish_recipe())

        recipe_agent = RecipeAgent(self._user_profile, full_recipe, recipe_title=recipe_title, user_id=self._user_id)
        ctx.session.update_agent(recipe_agent)
        return f"Switched to recipe mode for: {recipe_title}"


class RecipeAgent(Agent):
    def __init__(self, user_profile: dict, recipe: str, recipe_title: str = "", user_id: str = "default_user") -> None:
        super().__init__(
            instructions=RECIPE_PROMPT.format(
                voice_style=VOICE_STYLE,
                recipe=recipe,
                user_profile=_format_profile(user_profile),
                first_name=user_profile["first_name"],
            ),
        )
        self._user_profile = user_profile
        self._recipe_title = recipe_title
        self._user_id = user_id

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Strip old images, inject new video frames, and trigger summarization if needed."""
        _strip_old_images(turn_ctx)
        _inject_video_frames(turn_ctx, new_message)
        # Trigger background summarization on the agent's persistent chat context
        # (turn_ctx is a copy, so we check the real one)
        await _maybe_summarize(self.chat_ctx)

    @function_tool(
        description="Refresh the recipe card on the user's screen with updated title, ingredients, "
        "and/or steps. Call this whenever the user requests a substitution, quantity change, or any "
        "modification to the recipe. Pass the FULL updated title, ingredient list, and step list. "
        "The user's current step will be preserved."
    )
    async def update_recipe(
        self,
        ctx: RunContext,
        title: str,
        ingredients: str,
        steps: str,
    ) -> str:
        import re
        step_list = [s.strip() for s in re.split(r'\n\d+[\.\)]\s*', steps) if s.strip()]
        ingredient_list = [i.strip() for i in ingredients.strip().split('\n') if i.strip()]

        await _publish_data("recipe", {
            "type": "recipe_refresh",
            "title": title,
            "ingredients": ingredient_list,
            "steps": step_list,
        })
        return "Recipe card updated on the user's screen."

    @function_tool(
        description="Update the current step number on the user's screen. Call this EVERY time "
        "you start explaining a step — whether moving forward, backward, or repeating. "
        "Uses 1-based numbering (first step = 1)."
    )
    async def update_step(self, ctx: RunContext, step_number: int) -> str:
        await _publish_data("recipe", {
            "type": "step_update",
            "step_number": step_number,
        })
        return f"Step {step_number} is now highlighted on the user's screen."

    @function_tool(
        description="Request the user to enable their camera. Call this when the user says "
        "something that would benefit from visual context, like 'does this look done?', "
        "'what can I make with these?', 'look at this', etc."
    )
    async def request_camera(self, ctx: RunContext) -> str:
        await _publish_data("camera_request", {"type": "request_camera"})
        return "Camera request sent to the user's screen. Wait for them to enable it before commenting on visuals."

    @function_tool(
        description="Set a cooking timer. Use this proactively whenever a step involves waiting "
        "(e.g. simmering, baking, resting, marinating). The timer will alert the user when done."
    )
    async def set_timer(
        self,
        ctx: RunContext,
        label: str,
        duration_minutes: float,
    ) -> str:
        seconds = int(duration_minutes * 60)
        await _publish_timer(label, seconds)
        return f"Timer set: {label} for {duration_minutes} minutes."

    @function_tool(
        description="Show the user's grocery list on their screen. Call this when the user asks "
        "to see their grocery list or check what they need to buy."
    )
    async def show_grocery_list(self, ctx: RunContext) -> str:
        items = load_grocery_list(self._user_id)
        await _publish_data("grocery_list", {
            "type": "grocery_list_show",
            "items": items,
        })
        if not items:
            return "The grocery list is empty — nothing saved yet."
        total = sum(len(g["ingredients"]) for g in items)
        return f"Grocery list is now on screen with {total} items across {len(items)} recipe(s)."

    @function_tool(
        description="Save the current recipe's ingredients to the user's grocery list. Call this "
        "when the user wants to save ingredients for later shopping."
    )
    async def save_to_grocery_list(
        self,
        ctx: RunContext,
        recipe_title: str,
        ingredients: str,
    ) -> str:
        """ingredients should be a JSON array of ingredient strings."""
        import json as _json
        try:
            parsed = _json.loads(ingredients)
        except Exception:
            return "Failed to parse ingredients. Pass a valid JSON array of strings."
        items = load_grocery_list(self._user_id)
        items.append({"recipe": recipe_title, "ingredients": parsed, "checked": []})
        save_grocery_list(self._user_id, items)
        await _publish_data("grocery_list", {
            "type": "grocery_list_update",
            "items": items,
        })
        return f"Saved {len(parsed)} ingredients from {recipe_title} to the grocery list."

    @function_tool(
        description="Finish the current recipe and return to the main Chef Claude conversation. "
        "Call this when the user has completed the recipe or wants to stop cooking."
    )
    async def finish_recipe(self, ctx: RunContext) -> str:
        await _publish_data("recipe", {"type": "recipe_end"})

        # Save dish to history
        if self._recipe_title:
            from datetime import date
            profile = load_profile(self._user_id) or self._user_profile
            history = profile.get("dish_history", [])
            history.append({"title": self._recipe_title, "date": date.today().isoformat()})
            profile["dish_history"] = history
            save_profile(self._user_id, profile)
            self._user_profile.update(profile)

        chef = ChefAgent(self._user_profile, user_id=self._user_id)
        ctx.session.update_agent(chef)
        return "Recipe complete. Back to main conversation."


async def _capture_video_frames(room: rtc.Room) -> None:
    """Background task that captures the latest video frame from any participant's camera.

    Uses a two-task pattern: a fast reader drains the VideoStream into a single
    'latest frame' slot, and a slower sampler copies that frame into the ring
    buffer at ~2 FPS.  This ensures we always have the most recent frame
    regardless of stream throughput.
    """
    _latest_frame: dict[str, rtc.VideoFrame | None] = {"frame": None}

    while True:
        # Wait for a participant with a video track
        video_track: rtc.RemoteVideoTrack | None = None
        for p in room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track and pub.source == rtc.TrackSource.SOURCE_CAMERA:
                    video_track = pub.track  # type: ignore
                    break
            if video_track:
                break

        if video_track is None:
            _frame_buffer.clear()
            _latest_frame["frame"] = None
            await asyncio.sleep(1)
            continue

        stream: rtc.VideoStream | None = None
        reader_task: asyncio.Task | None = None
        try:
            stream = rtc.VideoStream(video_track)

            # Fast reader: drains every frame, always keeps the latest
            async def _drain():
                async for event in stream:
                    _latest_frame["frame"] = event.frame

            reader_task = asyncio.create_task(_drain())

            # Sampler: grab the latest frame at ~2 FPS into the ring buffer
            while not reader_task.done():
                if _latest_frame["frame"] is not None:
                    _frame_buffer.append(_latest_frame["frame"])
                await asyncio.sleep(0.5)

        except Exception:
            logger.debug("Video capture interrupted, restarting...")
        finally:
            if reader_task and not reader_task.done():
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            if stream is not None:
                await stream.aclose()
            _frame_buffer.clear()
            _latest_frame["frame"] = None
            await asyncio.sleep(1)


_encode_options = EncodeOptions(
    format="JPEG",
    quality=75,
    resize_options=ResizeOptions(width=1024, height=1024, strategy="scale_aspect_fit"),
)


async def _summarize_older_turns(items_to_summarize: list) -> str:
    """Use a fast cheap model to summarize older conversation turns."""
    # Extract text content from items for summarization
    lines = []
    for item in items_to_summarize:
        role = getattr(item, "role", "unknown")
        content = getattr(item, "content", "")
        if isinstance(content, list):
            text_parts = [c for c in content if isinstance(c, str)]
            content = " ".join(text_parts)
        if content:
            lines.append(f"{role}: {content}")

    if not lines:
        return ""

    conversation_text = "\n".join(lines)

    client = anthropic_sdk.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this cooking conversation into a compact paragraph. "
                "Preserve key facts: what's being cooked, current step, decisions made, "
                "user preferences mentioned, any issues encountered. Be concise.\n\n"
                f"{conversation_text}"
            ),
        }],
    )

    return response.content[0].text


async def _maybe_summarize(agent_chat_ctx: llm.ChatContext) -> None:
    """Check if we should trigger async summarization. Non-blocking."""
    global _summarization_in_progress

    items = agent_chat_ctx.items
    if len(items) <= SUMMARIZE_THRESHOLD or _summarization_in_progress:
        return

    _summarization_in_progress = True

    async def _do_summarize():
        global _summarization_in_progress
        try:
            async with _summarization_lock:
                items = agent_chat_ctx.items
                if len(items) <= SUMMARIZE_THRESHOLD:
                    return

                # Split: older items to summarize, recent items to keep
                split_point = len(items) - KEEP_RECENT_ITEMS
                older_items = items[:split_point]
                recent_items = items[split_point:]

                summary = await _summarize_older_turns(older_items)

                if summary:
                    # Replace older items with a single summary message
                    summary_msg = llm.ChatMessage(
                        role="assistant",
                        content=f"[Conversation so far: {summary}]",
                    )
                    items.clear()
                    items.append(summary_msg)
                    items.extend(recent_items)
                    logger.info(
                        "Compacted %d items into summary + %d recent items",
                        len(older_items), len(recent_items),
                    )
        except Exception:
            logger.exception("Error during async summarization")
        finally:
            _summarization_in_progress = False

    # Fire and forget — runs in background
    asyncio.create_task(_do_summarize())


def _strip_old_images(turn_ctx: llm.ChatContext) -> None:
    """Strip images from all but the most recent N messages."""
    items = turn_ctx.items
    if len(items) > KEEP_IMAGES_IN_LAST_N:
        for item in items[:-KEEP_IMAGES_IN_LAST_N]:
            if hasattr(item, "content") and isinstance(item.content, list):
                item.content = [c for c in item.content if not isinstance(c, llm.ImageContent)]


def _inject_video_frames(
    turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
) -> None:
    """If video frames are available, encode and inject them into the user message.
    Sends up to 3 recent frames so Claude gets temporal context (~1.5 sec window)."""
    if not _frame_buffer:
        return

    import base64

    frames = list(_frame_buffer)
    content = list(new_message.content) if isinstance(new_message.content, list) else [new_message.content]

    # Insert frames oldest-first before the text, so Claude sees the sequence
    for frame in frames:
        img_bytes = encode(frame, _encode_options)
        data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
        content.insert(len(content) - 1, llm.ImageContent(image=data_url))

    new_message.content = content


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    global _current_room
    _current_room = ctx.room

    # Start background video capture task
    asyncio.create_task(_capture_video_frames(ctx.room))

    # Use participant identity as user ID, fall back to a default
    user_id = "default_user"
    for p in ctx.room.remote_participants.values():
        user_id = p.identity
        break

    profile = load_profile(user_id) or {}

    if profile:
        agent = ChefAgent(profile, user_id=user_id)
    else:
        agent = OnboardingAgent(user_id, profile)

    tts_provider = os.environ.get("TTS_PROVIDER", "cartesia").lower()
    tts_voice = os.environ.get("TTS_VOICE_ID", "")
    if tts_provider == "elevenlabs":
        tts = elevenlabs.TTS(**({"voice_id": tts_voice} if tts_voice else {}))
    else:
        tts = cartesia.TTS(**({"voice": tts_voice} if tts_voice else {}))

    agent_session = AgentSession(
        stt=deepgram.STT(),
        llm=anthropic.LLM(model="claude-opus-4-6"),
        tts=tts,
        vad=silero.VAD.load(),
    )

    # Listen for UI selections (dish picks, etc.) and inject them as user messages
    @ctx.room.on("data_received")
    def _on_frontend_data(data: rtc.DataPacket):
        try:
            logger.info("Data received — topic=%s, from=%s", data.topic, getattr(data.participant, "identity", "unknown"))
            msg = json.loads(data.data.decode())
            if data.topic == "dish_selection" and msg.get("type") == "select_dish":
                title = msg.get("title", "")
                if title:
                    logger.info("Dish selected via UI: %s", title)
                    # Interrupt any current speech so the agent responds immediately
                    if agent_session.current_speech:
                        agent_session.interrupt()
                    agent_session.generate_reply(
                        user_input=f"I'd like to make {title}",
                    )
        except Exception:
            logger.exception("Error handling frontend data")

    await agent_session.start(
        agent=agent,
        room=ctx.room,
    )

    # Publish existing grocery list to frontend on connect
    grocery = load_grocery_list(user_id)
    if grocery:
        await _publish_data("grocery_list", {
            "type": "grocery_list_update",
            "items": grocery,
        })


if __name__ == "__main__":
    cli.run_app(server)
