#/Applications/anaconda3/envs/egg/bin/python
import argparse
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
from pdf2image import convert_from_path
from shutil import copyfile
from PIL import Image, ImageDraw, ImageFont
import re


def load_ventral(image_path, resize_factor):
    img = Image.open(image_path).convert('RGBA')
    img = img.rotate(270, expand=True)
    new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
    return img.resize(new_size, Image.LANCZOS)


def load_lateral(image_path, resize_factor):
    img = Image.open(image_path).convert('RGBA')
    new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
    return img.resize(new_size, Image.LANCZOS)


def load_colorbar(color_bar_path, resize_factor):
    img = Image.open(color_bar_path).convert('RGBA')
    new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
    img = img.resize(new_size, Image.LANCZOS)
    width, height = img.size
    crop_region = (int(width*(6/7)), 0, width, height)
    return img.crop(crop_region)


def draw_arrow(draw, start, end, arrow_size=10, fill=(0, 0, 0),
               width=20):
    """
    Draw an arrow on the canvas.
    
    Parameters:
        draw (ImageDraw.Draw): The drawing context.
        start (tuple): (x1, y1) coordinates of the start of the arrow.
        end (tuple): (x2, y2) coordinates of the end of the arrow.
        arrow_size (int): Size of the arrowhead.
        fill (tuple): Color of the arrow (R, G, B).
    """
    # Calculate the arrowhead points
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx**2 + dy**2)**0.5  # Length of the line
    if length == 0:
        return  # Avoid division by zero

    # Unit vector in the direction of the line
    unit_x = dx / length
    unit_y = dy / length

    # Perpendicular unit vector (for the arrowhead wings)
    perp_x = -unit_y
    perp_y = unit_x

        # Adjust the end point of the line to the back of the arrowhead
    adjusted_end = (
        end[0] - arrow_size * unit_x,
        end[1] - arrow_size * unit_y
    )

    # Draw the line (shaft of the arrow)
    draw.line([start, adjusted_end], fill=fill, width=width)

    # Arrowhead points
    arrow_left = (
        end[0] - arrow_size * unit_x + arrow_size * perp_x,
        end[1] - arrow_size * unit_y + arrow_size * perp_y
    )
    arrow_right = (
        end[0] - arrow_size * unit_x - arrow_size * perp_x,
        end[1] - arrow_size * unit_y - arrow_size * perp_y
    )

    # Draw the arrowhead (triangle)
    draw.polygon([end, arrow_left, arrow_right], fill=fill)


def plot_brain_baguette(ventrals, laterals, 
                        colorbar_path, 
                        outpath, 
                        lateral_resize=1.55, 
                        ventral_resize=2.25,
                        width_inches=7.5, height_inches=2.5,
                        dpi=600): 
    canvas_size = (int(width_inches*dpi), int(height_inches*dpi))
    brain_canvas = Image.new('RGBA', canvas_size, (255, 255, 255, 0))
    brain_canvas.info['dpi'] = (dpi, dpi)

    draw = ImageDraw.Draw(brain_canvas)
    font = ImageFont.truetype('/home/emcmaho7/.fonts/Roboto/Roboto-Regular.ttf', 80)

    lateral_pos = -50
    ventral_pos = int(height_inches*dpi*.05)
    time_pos = int(height_inches*dpi*.8)
    arrow_pos = int(height_inches*dpi*.75)
    title_pos = int(height_inches*dpi*.875)

    x = 35
    for i, (ventral, lateral) in enumerate(zip(ventrals, laterals)):
        # Open the images
        time = re.search(r"timems-(\d+)", ventral).group(1)
        ventral = load_ventral(ventral, ventral_resize)
        lateral = load_lateral(lateral, lateral_resize)

        # Paste the image onto the canvas
        brain_canvas.paste(ventral, (x, ventral_pos), mask=ventral)
        brain_canvas.paste(lateral, (x, lateral_pos), mask=lateral)

        if i == 0:
            arrow_start = x + int(ventral.width * .2)

        x += int(ventral.width * .5)
        draw.text((x, time_pos), font=font, text=time, fill=(0, 0, 0))

        if i+1 == len(ventrals):
            arrow_end = x + int(ventral.width * .25)
        
    # Make combined canvas
    combined_canvas = Image.new('RGB', canvas_size, (255, 255, 255))
    combined_canvas.info['dpi'] = (dpi, dpi)

    # Convert to 'RGB' to remove alpha channel and replace transparency with white
    combined_canvas.paste(brain_canvas, (0, 0), mask=brain_canvas)

    # Add the colorbar
    colorbar = load_colorbar(colorbar_path, ventral_resize)
    combined_canvas.paste(colorbar, (int(x*1.15), ventral_pos), mask=colorbar)

    # Add the time
    draw = ImageDraw.Draw(combined_canvas)
    font = ImageFont.truetype('/home/emcmaho7/.fonts/Roboto/Roboto-Regular.ttf', 110)
    draw.text((10, 10), font=font, text='C', fill=(0, 0, 0))

    # Draw the arrow
    draw_arrow(draw, start=(arrow_start, arrow_pos), 
               end=(arrow_end, arrow_pos),
               arrow_size=50, fill=(0, 0, 0),
               width=15)
    
    # Add time label
    font = ImageFont.truetype('/home/emcmaho7/.fonts/Roboto/Roboto-Regular.ttf', 100)
    draw.text((int(arrow_start + (arrow_end-arrow_start)/2), title_pos), font=font,
              text='Time (ms)', fill=(0, 0, 0))
    
    # Add rotated text "Prediction (r)" on the right side
    prediction_text = "Prediction (r)"
    rotated_canvas_size = (int(height_inches*dpi), int(width_inches*dpi))
    x_pos, y_pos = int(height_inches*dpi*0.29), 240
    text_image = Image.new('RGBA', rotated_canvas_size, (255, 255, 255, 0))  # Adjust size as needed
    text_draw = ImageDraw.Draw(text_image)
    font = ImageFont.truetype('/home/emcmaho7/.fonts/Roboto/Roboto-Regular.ttf', 80)
    text_draw.text((x_pos, y_pos), text=prediction_text, font=font, fill=(0, 0, 0))
    rotated_text = text_image.rotate(270, expand=True)
    combined_canvas.paste(rotated_text, (0, 0), mask=rotated_text)

    # Save image
    combined_canvas.save(outpath)


def combine_plots(out_file, img1, img2, 
                  width_inches=7.5, height_inches=5.5, dpi=600): 
    canvas_size = (int(width_inches*dpi), int(height_inches*dpi))
    canvas = Image.new('RGB', canvas_size, (255, 255, 255))
    canvas.info['dpi'] = (dpi, dpi)

    # Add the top plots
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (0, img1.height))
    canvas.save(out_file)


def get_files(search_str, times=[0,50],
              pattern=r"timems-(\d+)"):
    files = []
    for file in sorted(glob(search_str)):
        match = re.search(pattern, file)
        if match:
            time = int(match.group(1))
        else:
            print("No match found.")

        if time in times:
            files.append(file)
    return files


class PlotBaguette:
    def __init__(self, args):
        self.process = 'PlotBaguette'
        self.out_dir = args.out_dir 
        self.final_dir = args.final_dir
        self.brain_plots_path = f'{self.out_dir}/PlotWholeBrain'
        self.color_bar = f'{self.out_dir}/PlotWholeBrain/medial-rh/sub-02_time-080_timems-0.png'
        self.roi_plot = f'{self.out_dir}/PlotROIDecoding/roi_plot.pdf'
        self.out_plot = f'{self.final_dir}/Figure3.pdf'
        self.times = list(np.arange(args.start, args.stop+1, step=args.step))
        Path(f'{self.out_dir}/{self.process}').mkdir(exist_ok=True, parents=True)
        print(vars(self))

    def run(self):
        baguette_path = f'{self.out_dir}/{self.process}/baguette_plot.png'
        ventrals = get_files(f'{self.brain_plots_path}/ventral-rh/*png',
                             times=self.times)
        laterals = get_files(f'{self.brain_plots_path}/lateral-rh/*png',
                             times=self.times)
        plot_brain_baguette(ventrals, laterals, self.color_bar, baguette_path)

        roi_plot = convert_from_path(self.roi_plot, dpi=600)[0]
        baguette_plot = Image.open(baguette_path)
        
        combine_plots(f'{self.out_dir}/{self.process}/combined_plot.pdf',
                      roi_plot, baguette_plot)
        copyfile(f'{self.out_dir}/{self.process}/combined_plot.pdf', 
                 self.out_plot)


def main():
    parser = argparse.ArgumentParser(description='Plot the ROI regression results')
    parser.add_argument('--out_dir', '-o', type=str, help='directory for outputs',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/data/interim')
    parser.add_argument('--final_dir', type=str, help='experiment schematic',
                        default='/home/emcmaho7/scratch4-lisik3/emcmaho7/SIEEG_analysis/reports/figures/FinalFigures')
    parser.add_argument('--start', type=int, default=50, 
                        help='time to start plot')
    parser.add_argument('--stop', type=int, default=300, 
                        help='time to stop plot (inclusive)')
    parser.add_argument('--step', type=int, default=50, 
                        help='time to start plot')
    args = parser.parse_args()
    PlotBaguette(args).run()


if __name__ == '__main__':
    main()