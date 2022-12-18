import gradio as gr
import torch

from inference import convert_video
from model import MattingNetwork

# 检测cuda环境 如发现可用使用cuda加速
if torch.cuda.is_available():
    model_matting = MattingNetwork('mobilenetv3').eval().cuda()
else:
    model_matting = MattingNetwork('mobilenetv3').eval()

# 载入预训练好的视频去背景模型
model_matting.load_state_dict(torch.load('rvm_mobilenetv3.pth'))


def video_matting(videos):

    # 视频输出路劲
    output_file = '.\\output\\output.mp4'

    # 视频去背景
    convert_video(
        model_matting,  # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=videos,  # 视频文件，或图片序列文件夹
        output_type='video',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        output_composition=output_file,  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        output_video_mbps=4,  # 若导出视频，提供视频码率
        downsample_ratio=None,  # 下采样比，可根据具体视频调节，或 None 选择自动
        seq_chunk=4,  # 设置多帧并行计算
    )

    # 返回输出视频
    return output_file


def image_matting(image):

    # 视频输出路劲
    output_file = '.\\output\\output.png'

    # 视频去背景
    convert_video(
        model_matting,  # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=image,  # 视频文件，或图片序列文件夹
        output_type='png_sequence',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        output_composition=output_file,  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        downsample_ratio=None,  # 下采样比，可根据具体视频调节，或 None 选择自动
        seq_chunk=1,  # 设置多帧并行计算
    )

    # 返回输出视频
    return output_file


with gr.Blocks() as demo:

    # 设置网页描述
    gr.Markdown('''## Avatar: Easily Show Yourself  
        Please Chose User Model: **Avatar Pro** or **Avatar Lite**  
        In Avatar Program, You can easily generate a video with your own cartoon image for powerpoint presentation  
        In **Avatar Pro** mode, You need to upload a video and a powerpoint file  
        In **Avatar Lite** mode, You need to upload a selfie, a powerpoint and a presentation text''')

    with gr.Tab("Video Matting"):

        # 设置输入输出框GUI
        with gr.Row():
            video_input = gr.Video(label="Please Upload Your original Video")
            video_output = gr.Video(label="Here is your final Video")

        # 创造按钮
        video_matting_button = gr.Button("Remove!")

    with gr.Tab("Image Matting"):

        # 设置输入输出框GUI
        with gr.Row():
            image_input = gr.Image(label="Please Upload Your original Image")
            image_output = gr.Image(label="Here is your final Image")

        # 创造按钮
        image_matting_button = gr.Button("Remove!")

    # 联系输入输出
    video_matting_button.click(video_matting, inputs=video_input, outputs=video_output)
    image_matting_button.click(image_matting, inputs=image_input, outputs=image_output)

# 启动网咯
demo.launch()

