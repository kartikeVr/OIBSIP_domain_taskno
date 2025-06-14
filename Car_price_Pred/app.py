from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("/mnt/mydata/python_projects/python_projects/iris/Car_price_Pred/models/XGBRegressor.lb")

# Car name encoding
car_name_encoding = {
    'ritz': 0, 'sx4': 1, 'ciaz': 2, 'wagon r': 3, 'swift': 4, 'vitara brezza': 5,
    's cross': 7, 'alto 800': 10, 'ertiga': 13, 'dzire': 14, 'alto k10': 19, 'ignis': 20,
    '800': 36, 'baleno': 39, 'omni': 43, 'fortuner': 49, 'innova': 51, 'corolla altis': 54,
    'etios cross': 55, 'etios g': 57, 'etios liva': 64, 'corolla': 76, 'etios gd': 80,
    'camry': 84, 'land cruiser': 85, 'Royal Enfield Thunder 500': 98, 'UM Renegade Mojave': 99,
    'KTM RC200': 100, 'Bajaj Dominar 400': 101, 'Royal Enfield Classic 350': 102, 'KTM RC390': 103,
    'Hyosung GT250R': 104, 'Royal Enfield Thunder 350': 105, 'KTM 390 Duke ': 110,
    'Mahindra Mojo XT300': 111, 'Bajaj Pulsar RS200': 118, 'Royal Enfield Bullet 350': 120,
    'Royal Enfield Classic 500': 122, 'Bajaj Avenger 220': 124, 'Bajaj Avenger 150': 125,
    'Honda CB Hornet 160R': 126, 'Yamaha FZ S V 2.0': 127, 'Yamaha FZ 16': 129,
    'TVS Apache RTR 160': 132, 'Bajaj Pulsar 150': 133, 'Honda CBR 150': 134, 'Hero Extreme': 135,
    'Bajaj Avenger 220 dtsi': 137, 'Bajaj Avenger 150 street': 139, 'Yamaha FZ v 2.0': 140,
    'Bajaj Pulsar NS 200': 150, 'Bajaj Pulsar 220 F': 146, 'TVS Apache RTR 180': 148,
    'Hero Passion X pro': 149, 'Yamaha Fazer ': 152, 'Honda Activa 4G': 153,
    'TVS Sport ': 154, 'Honda Dream Yuga ': 156, 'Bajaj Avenger Street 220': 158,
    'Hero Splender iSmart': 162, 'Activa 3g': 163, 'Hero Passion Pro': 164,
    'Honda CB Trigger': 166, 'Yamaha FZ S ': 168, 'Bajaj Pulsar 135 LS': 170,
    'Activa 4g': 171, 'Honda CB Unicorn': 172, 'Hero Honda CBZ extreme': 173,
    'Honda Karizma': 174, 'Honda Activa 125': 175, 'TVS Jupyter': 176,
    'Hero Honda Passion Pro': 178, 'Hero Splender Plus': 179, 'Honda CB Shine': 180,
    'Bajaj Discover 100': 181, 'Suzuki Access 125': 183, 'TVS Wego': 184,
    'Honda CB twister': 185, 'Hero Glamour': 186, 'Hero Super Splendor': 187,
    'Bajaj Discover 125': 189, 'Hero Hunk': 190, 'Hero Ignitor Disc': 191,
    'Hero CBZ Xtreme': 192, 'Bajaj ct 100': 193, 'i20': 199, 'grand i10': 200,
    'i10': 201, 'eon': 202, 'xcent': 204, 'elantra': 209, 'creta': 210,
    'verna': 213, 'city': 249, 'brio': 250, 'amaze': 257, 'jazz': 261
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        car_name = request.form.get('car_name').strip().lower()
        year = int(request.form.get('year'))
        present_price = float(request.form.get('present_price'))
        driven_kms = int(request.form.get('driven_kms'))
        fuel_type = request.form.get('fuel_type')
        selling_type = request.form.get('selling_type')
        transmission = request.form.get('transmission')
        owner = int(request.form.get('owner'))

        # Encode inputs manually
        if car_name not in car_name_encoding:
            raise ValueError(f"Unknown car name: '{car_name}'. Please enter a valid car name.")

        car_name_encoded = car_name_encoding[car_name]
        fuel_encoded = 0 if fuel_type == 'Diesel' else 1
        selling_encoded = 1 if selling_type == 'Dealer' else 0
        transmission_encoded = 1 if transmission == 'Manual' else 0

        # Create input array for prediction
        features = np.array([[car_name_encoded, year, present_price, driven_kms,
                              fuel_encoded, selling_encoded, transmission_encoded, owner]])

        # Predict and format result
        predicted_price = round(model.predict(features)[0], 2)

        return render_template('index.html',
                               prediction_text=f'Estimated Selling Price: ₹{predicted_price} Lakhs',
                               show_result=True,
                               car_name=car_name.title())

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}',
                               show_result=True)

if __name__ == '__main__':
    app.run(debug=True)
