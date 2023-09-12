import gradio as gr
import pandas as pd

pd.set_option('display.max_colwidth', None)
from parsernaam.parse import ParseNames

def parse_names(names):
    given_names = names.split(",")
    df = pd.DataFrame({'name': given_names})
    df = ParseNames.parse(df)
    print(df)
    output = ""
    for parsed_name in df['parsed_name']:
        for name_dict in parsed_name:
            name = name_dict['name']
            name_type = name_dict['type']
            prob = name_dict['prob']
            output += f"{name} (type: {name_type}, score: {prob:.2f})\n"
        output += "\n"
    return output

iface = gr.Interface(
    fn=parse_names,
    inputs=gr.components.Textbox(lines=10, label="Names"),
    outputs=gr.components.Textbox(lines=20, label="Parsed names"),
    title="Parse names",
    description="Parse names",
    allow_flagging="never",
    examples=[
        ["Jan Petersen", "Piet", "Janssen"],
        ["Jan", "Piet", "Janssen", "Petersen", "Jansen", "Pietersen"],
    ],
)

if __name__ == "__main__":
    iface.launch()
