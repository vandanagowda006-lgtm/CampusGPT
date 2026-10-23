import re
import html
import streamlit as st
import streamlit.components.v1 as components


def render_answer(answer):
    """Display Markdown answers and render Mermaid diagrams inline."""
    if not isinstance(answer, str):
        st.write(answer)
        return

    pattern = r"```mermaid\s*\n(.*?)```"
    matches = list(re.finditer(pattern, answer, flags=re.IGNORECASE | re.DOTALL))

    if not matches:
        st.markdown(answer)
        return

    last_end = 0

    for match in matches:
        preceding_text = answer[last_end:match.start()].strip()
        if preceding_text:
            st.markdown(preceding_text)

        mermaid_code = match.group(1).strip()
        escaped_code = html.escape(mermaid_code)

        diagram_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
body {{ margin: 0; padding: 12px; font-family: Arial, sans-serif; background: transparent; }}
.diagram {{ width: 100%; overflow-x: auto; background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; box-sizing: border-box; }}
.mermaid {{ display: flex; justify-content: center; min-width: 300px; }}
.error {{ color: #b91c1c; white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="diagram"><pre class="mermaid">{escaped_code}</pre></div>
<script>
if (window.mermaid) {{
    mermaid.initialize({{ startOnLoad: false, theme: "default", securityLevel: "strict" }});
    mermaid.run().catch(function(error) {{
        document.body.insertAdjacentHTML("beforeend", '<p class="error">The diagram could not be rendered. Check its Mermaid syntax.</p>');
        console.error(error);
    }});
}} else {{
    document.body.insertAdjacentHTML("beforeend", '<p class="error">Mermaid could not load. Check your internet connection.</p>');
}}
</script>
</body>
</html>"""

        components.html(diagram_html, height=380, scrolling=True)
        last_end = match.end()

    remaining_text = answer[last_end:].strip()
    if remaining_text:
        st.markdown(remaining_text)
