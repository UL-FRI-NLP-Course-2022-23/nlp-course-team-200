from PIL import Image, ImageDraw, ImageFont
import os

def visualize(r, fable_sentiments):

    image_width = 600 
    image_height = 300
    line_thickness = 3
    offset = 2

    font = ImageFont.truetype("arial", 16)

    characters_sentiments = {c: [[f[1], f[2]] for f in fable_sentiments if f[0] == c] for c in set([f[0] for f in fable_sentiments])}

    for characters, sentiments in characters_sentiments.items():

        y = image_height // 2
        x = image_width // 3

        c1, c2 = characters
        n_sentiments = len(sentiments)

        image = Image.new("RGB", (image_width, image_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        length_sum = sum(s[1] for s in sentiments)

        for sent, sentence_length in sentiments:
            label, score = sent
            if label == 'POSITIVE':
                color = (0, int(score * 255), 0)
            else:
                color = (int(score * 255), 0, 0)

            line_length = int((image_width // 3) * (sentence_length / length_sum))

            draw.line([(x + offset, y), (x + line_length - offset, y)], fill=color, width=line_thickness)

            x += line_length

        draw.text(((image_width // 3) - (len(c1) * 10), y - 10), c1, fill=(0,0,0), font=font)
        draw.text((x + 10, y - 10), c2, fill=(0,0,0), font=font)

        # Save the visualization as a PNG image
        image.save(os.path.join("results", "visualization", f"{r.title}_{c1}_{c2}.png"))