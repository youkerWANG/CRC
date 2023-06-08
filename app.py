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

max_num = 0
image_dir = app.config['UPLOAD_FOLDER']
@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/upload_photos', methods=["POST"])
def photos():
    global max_num
    filenames = os.listdir(image_dir)
    if filenames:
        max_num = max(int(name.split('.')[0]) for name in filenames)
    max_num += 1

    files = request.files.getlist('photo')
    for i in files:
        filename = str(max_num) + '.jpg'
        filepath = os.path.join(image_dir, filename)
        i.save(filepath)
        max_num += 1  # 更新序号

    return redirect(url_for('index', _anchor='upupload'))


@app.route('/upload_option', methods=['POST', 'GET'])
def options():
    outputs = []

    elected_options = request.form.getlist('options') # 옵션
    directory = 'static/user_photos/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filenames = os.listdir(directory)
    photo_paths = []
    max_num = 0
    for filename in filenames:
        if filename.endswith('.jpg'):
            num = int(filename.split('.')[0])
            if num > max_num:
                max_num = num
                photo_paths = [os.path.join('static/user_photos', filename)]
            elif num == max_num:
                photo_paths.append(os.path.join('static/user_photos', filename))

    img_path = photo_paths[-1]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    img_input = img.astype(float) / 255.
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

    area_scratch = (outputs_scratch[0]/40).sum()
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

    area_Seperated = (outputs_Seperated[0]/40).sum()
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

    area_crushed = (outputs_crushed[0]/40).sum()
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

    area_Breakage = (outputs_Breakage[0]/40).sum()
    total_Breakage += area_Breakage * price_Breakage
    print(area_Breakage)
    print(total_Breakage)
    damage = [0] * 4
    damage[0] = area_Breakage
    damage[1] = area_crushed
    damage[2] = area_scratch
    damage[3] =area_Seperated

    part_change_price = {
        'part01': 540000,  # 1.앞범퍼
        "part02": 530000,  # 2.뒷범퍼
        "part03": 770000,  # 3.트렁크
        "part04": 210000,  # 4.전조등
        "part05": 150000,  # 5.후램프
        "part06": 860000,  # 6.본넷
        "part07": 290000,  # 7.앞유리
        "part08": 2150000,  # 8.루프
        "part09": 210000,  # 9.뒷유리
        "part10": 710000,  # 10.앞펜더
        "part11": 2100000,  # 11.뒷펜더
        "part12": 160000,  # 12.사이드미러
        "part13": 800000,  # 13.앞도어
        "part14": 850000,  # 14.뒷도어
        "part15": 140000,  # 15.스텝
        "part16": 100000,  # 16.타이어
        "part17": 170000  # 17.바퀴 휠
    }

    # 사용자로부터 부품 번호 입력 받기
    part_name = ''.join(request.form.getlist('car_part'))

    def damage_repair_cost(damage, part_name):
        sheet_metal_cost = calculate_sheet_metal_cost(damage, part_name)
        paint_cost = calculate_paint_cost(damage, part_name)

        if damage[3] >= 50:  # 이격 정도에 따른 교체 여부 판단
            return part_change_price[part_name]  # 교체 비용
        else:
            return min(part_change_price[part_name], (sheet_metal_cost + paint_cost))  # 미교체시 판금 도색 비용 반환

    def calculate_sheet_metal_cost(damage, part_name):  # 판금
        if damage[0] < 20 and damage[1] < 20:  # damage[0] : 찢어짐 , damage[1] : 찌그러짐
            return 0  # 찢어짐 일정 수치 미만 무시

        elif damage[0] + damage[1] > 600:
            return (part_change_price[part_name] - calculate_paint_cost(damage, part_name))

        elif damage[0] > 300 or damage[1] > 400:
            return (part_change_price[part_name] - calculate_paint_cost(damage, part_name))

        elif damage[0] >= 20 and damage[1] >= 20:
            return (damage[0] + damage[1]) / 6 * 10000

        elif damage[0] >= 20 and damage[1] < 20:
            return (damage[0] / 3) * 20000

        elif damage[1] >= 20 and damage[0] < 20:
            if damage[2] >= 50:
                return (damage[1]) / 3 * 10000
            else:
                return 50000  # 판금 비용
        else:
            return 0

    def calculate_paint_cost(damage, part_name):
        basic_paint = 300000
        if damage[2] < 20:
            return 0.0

        elif damage[2] >= 20 and damage[2] <= 60:
            return damage[2] * 5000

        elif part_name in ["본넷", "루프", "앞도어", "뒷도어"]:
            if part_name == "본넷":  # 본넷
                return basic_paint + 100000
            elif part_name == "루프":  # 루프
                return basic_paint + 200000
            elif part_name in ["앞도어", "뒷도어"]:  # 도어
                return basic_paint + 50000
        else:
            return basic_paint

    # 수리 비용 계산
    repair_cost = damage_repair_cost(damage, part_name)
    rounded_repair_cost = round(repair_cost, -4)  # 100000자리 이하 반올림
    rounded_repair_cost_int = int(rounded_repair_cost)  # 정수형으로 변환
    repair_method1 = "부품 교체"
    repair_method2 = "판금 및 도색"
    if repair_cost == part_change_price[part_name]:
        repair_method = repair_method1
    else:
        repair_method = repair_method2

    context = {
        'repair_method': repair_method,
        'rounded_repair_cost': rounded_repair_cost_int
    }

    return render_template('result.html', **context)

    # print("수리 비용: {} 원".format(rounded_repair_cost_int))


    # total_price = total_Breakage + total_crushed + total_Seperated + total_scratch
    # context = {'scratch_area': area_scratch, 'scratch_price': total_scratch, 'seperated_area': area_Seperated,
    #            'sep_price': total_Seperated,
    #            'crushed_area': area_crushed, 'crushed_price': total_crushed, 'breakage_area': area_Breakage,
    #            'bre_price': total_Breakage,
    #            'total_price': total_price}
    # return render_template('result.html', **context)
    #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
