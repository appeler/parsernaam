"""Gradio web interface for interactively parsing names."""

import gradio as gr
import pandas as pd

from parsernaam.parse import ParseNames


def parse_names(names: str) -> str:
    """Parse a comma-separated string of names and format the results.

    Args:
        names: Comma-separated names to parse.

    Returns:
        Human-readable summary of each parsed name.
    """
    given_names = [name.strip() for name in names.split(",")]
    df = pd.DataFrame({"name": given_names})
    df = ParseNames.parse(df)
    output = []
    for parsed_name in df["parsed_name"]:
        name = parsed_name["name"]
        name_type = parsed_name["type"]
        prob = parsed_name["prob"]
        output.append(f"{name} (type: {name_type}, score: {prob:.2f})")
    return "\n".join(output)


iface = gr.Interface(
    fn=parse_names,
    inputs=gr.components.Textbox(lines=10, label="Names"),
    outputs=gr.components.Textbox(lines=20, label="Parsed names"),
    title="Parse names",
    description="Explore model-based first/last ordering labels for name strings.",
    flagging_mode="never",
    examples=[
        ["Jan Petersen, Piet, Janssen"],
        ["Jan, Piet, Janssen, Petersen, Jansen, Pietersen"],
    ],
)

if __name__ == "__main__":
    iface.launch()
