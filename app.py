from flask import Flask, render_template, request
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)


def predict_label(img_path):
	model = load_model("model/model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]


@app.route("/")
def main():
	return render_template("ocular.html")


@app.route("/predictOcular", methods = ['GET', 'POST'])
def get_output():
	dic ={ 0:"No chance of disease", 1:"chance Of Ocular disease!"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)
		print(round(p[0]))

	return render_template("ocular.html", prediction = dic[round(p[0])], img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)
