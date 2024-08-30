from final import main
import gradio as gr
import time
# Path to your GIF
gif_path = "giphy.webp"

def display_gif():
    return gif_path
def call(language):
    # This function will be called when the user submits the form
    dic={"Kannada":"1","Hindi":"2","English":"3"}
    option=dic[language]
    main(option)



with gr.Blocks() as demo:
    with gr.Column() as column:
        language = gr.Radio(choices=["English", "Hindi", "Kannada"], label="Language")
        submit_button = gr.Button("Submit")
        gif_output = gr.Image(type="filepath", elem_id="listening_gif", visible=False)

    # Define the action for the submit button
    submit_button.click(fn=display_gif,outputs=gif_output).then(fn=call,inputs=language)

    # CSS to style the GIF
    demo.css = """
    #listening_gif {
        display: flex;
        justify-content: center;
        align-items: center;
        width: auto;
        height: auto;
    }

    #listening_gif > div > img {
        display: block;
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
    }
    """

demo.launch()

