import os
import shutil
import sys
import gradio as gr
from src.gradio_demo import SadTalker

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath", elem_id="img2img_image").style(width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")

                        if sys.platform != 'win32' and not in_webui:
                            from src.utils.text2speech import TTSTalker
                            tts_talker = TTSTalker()
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])

                with gr.Tabs(elem_id="sadtalker_batch_mode"):
                    with gr.TabItem('Batch Mode'):
                        batch_image_path = gr.Textbox(label="Image Path", placeholder="Enter the path for the image")
                        codeword_image_pairs = gr.Textbox(label="Code Word and Image Paths", lines=5, placeholder="Enter new-line delimited code word and image path pairs, separated by commas")
                        batch_audio_paths = gr.Textbox(label="Audio Paths", lines=5, placeholder="Enter new-line delimited paths for audio files")

            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown("need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials")
                        with gr.Column(variant='panel'):
                            mode = gr.Radio(['Single Mode', 'Batch Mode'], value='Single Mode', label='Mode')
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0)
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?")
                            preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='full', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)", value=True)
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2)
                            expression_scale = gr.Slider(label="expression scale", step=0.1, minimum=0, maximum=2, value=1.0)
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

        def process_single(source_image, driven_audio, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale):
            return sad_talker.test(source_image, driven_audio, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale)


        def process_batch(image_path, audio_paths, codeword_image_pairs, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale):
            # Strip double quotes
            image_path = image_path.replace('"', '')
            audio_paths = audio_paths.replace('"', '')
            codeword_image_pairs = codeword_image_pairs.replace('"', '')

            audio_paths_list = audio_paths.split('\n')
            codeword_image_dict = {}
            for pair in codeword_image_pairs.split('\n'):
                if ',' in pair:
                    codeword, img_path = pair.split(',', 1)
                    codeword_image_dict[codeword] = img_path

            videos = []

            working_dir = './working_dir'
            os.makedirs(working_dir, exist_ok=True)

            for audio_path in audio_paths_list:
                # Determine the image to use based on the code word
                selected_image_path = image_path
                for codeword, img_path in codeword_image_dict.items():
                    if codeword in audio_path:
                        selected_image_path = img_path
                        break

                working_image_path = os.path.join(working_dir, os.path.basename(selected_image_path))
                shutil.copy(selected_image_path, working_image_path)
                working_audio_path = os.path.join(working_dir, os.path.basename(audio_path))
                shutil.copy(audio_path, working_audio_path)

                try:
                    video = sad_talker.test(working_image_path, working_audio_path, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale)
                    videos.append(video)
                except Exception as e:
                    print(f"Error processing audio file {working_audio_path} with image {working_image_path}: {e}")
                    videos.append(None)
                finally:
                    print(f"Cleaning up working audio {working_audio_path}")
                    os.remove(working_audio_path)

            return videos

        def handle_submit(mode, source_image, driven_audio, batch_image_path, batch_audio_paths, codeword_image_pairs, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale):
            if mode == 'Single Mode':
                return process_single(source_image, driven_audio, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale)
            else:
                return process_batch(batch_image_path, batch_audio_paths, codeword_image_pairs, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale)


        submit.click(
            fn=handle_submit,
            inputs=[mode, source_image, driven_audio, batch_image_path, batch_audio_paths, codeword_image_pairs, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image, pose_style, expression_scale],
            outputs=[gen_video]
        )

    return sadtalker_interface

if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    demo.launch(share=True)