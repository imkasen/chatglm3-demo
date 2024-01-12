"""
UI 界面
"""
import gradio as gr

from .ui_functions import clear_messages, llm_reply, llm_stream_reply, query_user_input

# Gradio UI
with gr.Blocks(title="ChatGLM3-6B Gradio Simple Demo") as demo:
    gr.HTML(value="""<h1 align="center">ChatGLM3-6B Gradio Simple Demo</h1>""")

    url_text = gr.Textbox(
        label="API 地址",
        placeholder="输入后端 API 地址",
        value="http://127.0.0.1:8000",
        interactive=True,
    )

    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder="Input...", lines=5, container=False)
            submit_btn = gr.Button(value="Submit", variant="primary")
        with gr.Column(scale=1):
            empty_btn = gr.ClearButton(value="Clear History", size="sm")

            top_p_input = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.8,
                step=0.01,
                label="Top P",
                interactive=True,
            )
            temperature_input = gr.Slider(
                minimum=0.01,
                maximum=1,
                value=0.6,
                step=0.01,
                label="Temperature",
                interactive=True,
            )

    empty_btn.add(components=[user_input, chatbot])
    empty_btn.click(  # pylint: disable=E1101
        fn=clear_messages,
        inputs=url_text,
        outputs=None,
    )

    submit_btn.click(  # pylint: disable=E1101
        fn=query_user_input,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
        queue=False,
    ).then(
        # fn=llm_reply,
        fn=llm_stream_reply,
        inputs=[url_text, chatbot, top_p_input, temperature_input],
        outputs=chatbot,
    )
