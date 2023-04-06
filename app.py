from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/user_photos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


@app.route('/upload_photos', methods=["POST", "GET"])
def photos():
    files = request.files.getlist('photo')

    for i in files:
        filename = i.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        i.save(filepath)

    return redirect(url_for('index', _anchor='upupload'))


@app.route('/upload_option', methods=['POST'])
def options():
    elected_options = request.form.getlist('options')
    print(elected_options)
    return redirect(url_for('index', _anchor='upupload'))


if __name__ == '__main__':
    app.run()
