import html
import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("FINDIT_API", "http://127.0.0.1:8000").rstrip("/")
# Keep in sync with app/config.py.
LOW_CONFIDENCE = float(os.environ.get("FINDIT_LOW_CONF", 0.2))
# Raw CLIP cosine scores are not probabilities; this maps them to a 0-100 display scale,
# to be calibrated on the test set (M5).
SCORE_FLOOR = float(os.environ.get("FINDIT_SCORE_FLOOR", 0.15))
SCORE_CEIL = float(os.environ.get("FINDIT_SCORE_CEIL", 0.35))
UNREACHABLE = "API not reachable. Start it with: uvicorn app.main:app"
STYLE = (Path(__file__).with_name("style.css")).read_text(encoding="utf-8")

st.set_page_config(page_title="FindIt AI", page_icon="📍", layout="wide")
st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)


def to_percent(score):
    return round(100 * min(1.0, max(0.0, (score - SCORE_FLOOR) / (SCORE_CEIL - SCORE_FLOOR))))


def friendly_error(exc):
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = None
        return str(detail) if detail else f"Request failed ({response.status_code})."
    return UNREACHABLE


def api_get(path):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=20)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, friendly_error(exc)


def api_post(path, **kwargs):
    try:
        response = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, friendly_error(exc)


def api_patch(path, **kwargs):
    try:
        response = requests.patch(f"{API_BASE}{path}", timeout=20, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, friendly_error(exc)


def image_src(item):
    url = item.get("image_url") or ""
    if url.startswith(("http://", "https://")):
        return url
    return f"{API_BASE}{url}"


def render_card(item, score=None):
    if score is None:
        pill = ""
    else:
        title = html.escape(f"cosine {score:.3f}", quote=True)
        pill = f'<div class="pill" title="{title}">{to_percent(score)}% match</div>'
    place = html.escape(str(item.get("place") or "Unspecified spot"))
    note = html.escape(str(item.get("note") or "No details shared yet."))
    src = html.escape(image_src(item), quote=True)
    st.markdown(
        f'<div class="card"><img class="card-img" src="{src}" alt="Found item">'
        f'{pill}<div class="card-place">{place}</div>'
        f'<div class="card-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_grid(items, with_scores=False):
    for start in range(0, len(items), 3):
        columns = st.columns(3)
        for column, item in zip(columns, items[start:start + 3]):
            with column:
                render_card(item, item.get("score") if with_scores else None)


def sidebar():
    st.sidebar.markdown(
        '<div class="brand"><span class="brand-pin">📍</span><div>'
        '<div class="brand-name">FindIt AI</div>'
        '<div class="brand-tag">LOST · FOUND · BACK</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="side-label">Your campus</div>', unsafe_allow_html=True)
    pages = [("search", "Explore board"), ("report", "Report found"), ("admin", "Admin view")]
    for key, label in pages:
        kind = "primary" if st.session_state.get("page") == key else "secondary"
        if st.sidebar.button(label, key=f"nav_{key}", type=kind, width="stretch"):
            if st.session_state.get("page") != key:
                st.session_state.page = key
                st.rerun()
    st.sidebar.markdown(
        '<div class="side-card">Photo-match hints: matches use CLIP '
        'text-to-image similarity.</div>',
        unsafe_allow_html=True,
    )


def explore():
    items, error = api_get("/items")
    items = items or []
    unclaimed = [item for item in items if item.get("status") != "claimed"]
    claimed = [item for item in items if item.get("status") == "claimed"]
    places = sorted({item.get("place") for item in items if item.get("place")})

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown(
            '<div class="board-hero"><div class="rule"></div>'
            '<div class="eyebrow">Campus lost &amp; found · live board</div>'
            "<h1>Describe it.<br><span class=\"lime\">We'll find it.</span></h1>"
            '<p>A short description is all it takes to search the found items '
            "logged across campus.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="stats-card"><div class="num">{len(items)} '
            f'<span>active reports</span></div><div class="sub">'
            f"{len(unclaimed)} unclaimed · {len(claimed)} claimed</div></div>",
            unsafe_allow_html=True,
        )
    if error:
        st.error(error)

    with st.container(border=True):
        description = st.text_area(
            "What did you lose?",
            placeholder="e.g. a black tote bag with a silver clasp",
            height=90,
        )
        spot = st.selectbox("Campus spot", ["Everywhere"] + places)
        k = st.slider(
            "Number of results to show",
            1,
            10,
            5,
            help="Maximum number of top matches returned (1-10)",
        )
        if st.button("Search lost items", type="primary", width="stretch"):
            if not description.strip():
                st.warning("Add a short description before searching.")
            else:
                payload = {"description": description.strip(), "k": k}
                if spot != "Everywhere":
                    payload["place"] = spot
                results, search_error = api_post("/search", json=payload)
                if search_error:
                    st.error(search_error)
                else:
                    st.session_state.search_results = results or []
                    st.session_state.searched = True

    results = st.session_state.get("search_results") or []
    if st.session_state.get("searched"):
        if results and max(item["score"] for item in results) < LOW_CONFIDENCE:
            st.warning("No confident match. Try more detail or check back later.")
        st.markdown("### Search results")
        if not results:
            st.info("No found items matched this search yet.")
        else:
            render_grid(results, with_scores=True)
    else:
        latest = sorted(unclaimed, key=lambda item: item.get("item_id", 0), reverse=True)
        st.markdown("### Around campus")
        st.markdown('<p class="muted">The latest unclaimed items logged by finders.</p>',
                    unsafe_allow_html=True)
        if not latest:
            st.info("No unclaimed items on the board yet.")
        else:
            render_grid(latest[:9])


def report():
    st.markdown("## Report a found item")
    st.markdown('<p class="muted">Snap it, place it, and put it back in reach.</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Photo of the item", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        st.image(uploaded, caption="Preview", width="stretch")
    place = st.text_input("Where did you find it?", placeholder="e.g. Library, 2nd floor")
    note = st.text_area("Note", placeholder="Colour, brand, distinguishing details…")
    if st.button("Post found item", type="primary"):
        if uploaded is None:
            st.warning("Add a photo of the item before posting.")
        else:
            files = {
                "image": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "image/jpeg",
                )
            }
            created, error = api_post(
                "/found", files=files, data={"place": place.strip(), "note": note.strip()}
            )
            if error:
                st.error(error)
            else:
                st.success(f"Posted! Report #{created.get('item_id')} is on the board.")


def admin():
    st.markdown("## Admin view")
    st.markdown('<p class="muted">Review reports and mark items once they are claimed.</p>',
                unsafe_allow_html=True)
    items, error = api_get("/items")
    if error:
        st.error(error)
        return
    if not items:
        st.info("No reports yet.")
        return
    for item in sorted(items, key=lambda row: row.get("item_id", 0), reverse=True):
        status = item.get("status") or "unclaimed"
        col_img, col_id, col_note, col_meta, col_action = st.columns([1, 2, 2.5, 1.6, 1.2])
        with col_img:
            st.image(image_src(item), width="stretch")
        with col_id:
            st.markdown(f"**#{item.get('item_id')}**")
            st.markdown(html.escape(str(item.get("place") or "No spot given")))
        with col_note:
            st.markdown(html.escape(str(item.get("note") or "No note.")))
        with col_meta:
            st.markdown(html.escape(status))
            st.caption(html.escape(str(item.get("found_on") or "")))
        with col_action:
            if status != "claimed" and st.button(
                "Mark claimed", key=f"claim_{item.get('item_id')}"
            ):
                _, patch_error = api_patch(
                    f"/items/{item.get('item_id')}/status", json={"status": "claimed"}
                )
                if patch_error:
                    st.error(patch_error)
                else:
                    st.rerun()


if "page" not in st.session_state:
    st.session_state.page = "search"
sidebar()
{"search": explore, "report": report, "admin": admin}[st.session_state.page]()
