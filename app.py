from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
from Models import Unet

app = Flask(__name__)

UPLOAD_FOLDER = 'static/user_photos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight_scratch = 'static/Modules/[DAMAGE][Scratch_0]Unet.pt'
weight_Seperated = 'static/Modules/[DAMAGE][Seperated_1]Unet.pt'
weight_crushed = 'static/Modules/[DAMAGE][Crushed_2]Unet.pt'
weight_Breakage = 'static/Modules/[DAMAGE][Breakage_3]Unet.pt'
# scratch_model
model_scratch = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model_scratch.model.load_state_dict(torch.load(weight_scratch, map_location=torch.device(device)))
model_scratch.eval()
# Seperated_model
model_Seperated = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model_Seperated.model.load_state_dict(torch.load(weight_Seperated, map_location=torch.device(device)))
model_Seperated.eval()
# crushed_model
model_crushed = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model_crushed.model.load_state_dict(torch.load(weight_crushed, map_location=torch.device(device)))
model_crushed.eval()
# Breakage_model
model_Breakage = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model_Breakage.model.load_state_dict(torch.load(weight_Breakage, map_location=torch.device(device)))
model_Breakage.eval()


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/upload_photos', methods=["POST"])
def photos():
    files = request.files.getlist('photo')

    for i in files:
        filename = i.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        i.save(filepath)

    return redirect(url_for('index', _anchor='upupload'))


@app.route('/upload_option', methods=['POST', 'GET'])
def options():
    elected_options = request.form.getlist('options')
    directory = 'static/user_photos/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    photo_paths = [os.path.join('static/user_photos', filename) for filename in os.listdir(directory) if
                   filename.endswith('.jpg')]

    img_path = photo_paths[-1]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    img_input = img / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float().to(device)
    img_input = img_input.unsqueeze(0)

    # scratch_outputs
    output_scratch = model_scratch(img_input)

    img_output = torch.argmax(output_scratch, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs_scratch = [img_output]

    price_scratch = 50  # Scratch_0

    total_scratch = 0

    area_scratch = outputs_scratch[0].sum()
    total_scratch += area_scratch * price_scratch
    print(area_scratch)
    print(total_scratch)

    # Seperated_outputs
    output_Seperated = model_Seperated(img_input)

    img_output = torch.argmax(output_Seperated, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs_Seperated = [img_output]

    price_Seperated = 60  # Scratch_0

    total_Seperated = 0

    area_Seperated = outputs_Seperated[0].sum()
    total_Seperated += area_Seperated * price_Seperated

    print(area_Seperated)
    print(total_Seperated)
    # crushed_output
    output_Crushed = model_crushed(img_input)

    img_output = torch.argmax(output_Crushed, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs_crushed = [img_output]

    price_crushed = 70  # Scratch_0

    total_crushed = 0

    area_crushed = outputs_crushed[0].sum()
    total_crushed += area_crushed * price_crushed

    print(area_crushed)
    print(total_crushed)
    # breakage model

    output_Breakage = model_Breakage(img_input)

    img_output = torch.argmax(output_Breakage, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs_Breakage = [img_output]

    price_Breakage = 80  # Scratch_0

    total_Breakage = 0

    area_Breakage = outputs_Breakage[0].sum()
    total_Breakage += area_Breakage * price_Breakage
    print(area_Breakage)
    print(total_Breakage)

    total_price = total_Breakage + total_crushed + total_Seperated + total_scratch
    context = {'scratch_area': area_scratch, 'scratch_price': total_scratch, 'seperated_area': area_Seperated,
               'sep_price': total_Seperated,
               'crushed_area': area_crushed, 'crushed_price': total_crushed, 'breakage_area': area_Breakage,
               'bre_price': total_Breakage,
               'total_price': total_price}
    return render_template('result.html', **context)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
