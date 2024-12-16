from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

def f1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1

app = Flask(__name__)

CORS(app, supports_credentials=True)

model = tf.keras.models.load_model(
    "my_model-2.keras", 
    custom_objects={"f1_score": f1_score}
)

target_categorical_columns = [
    'proto_arp', 'proto_ospf', 'proto_other', 'proto_sctp', 'proto_tcp', 'proto_udp', 'proto_unas',
    'service_-', 'service_dhcp', 'service_dns', 'service_ftp', 'service_ftp-data', 'service_http',
    'service_irc', 'service_pop3', 'service_radius', 'service_smtp', 'service_snmp', 'service_ssh',
    'service_ssl', 'state_CON', 'state_ECO', 'state_FIN', 'state_INT', 'state_PAR', 'state_REQ',
    'state_RST', 'state_URN', 'state_no'
]

def preprocess_features(features):
    try:
        feature_df = pd.DataFrame([features])

        allowed_protos = ['tcp', 'udp', 'unas', 'arp', 'ospf', 'sctp']
        feature_df['proto'] = feature_df['proto'].apply(lambda x: x if x in allowed_protos else 'other')

        categorical_features = ['proto', 'service', 'state']
        categorical_encoded = pd.get_dummies(feature_df[categorical_features], drop_first=True)

        for col in target_categorical_columns:
            if col not in categorical_encoded:
                categorical_encoded[col] = 0

        categorical_encoded = categorical_encoded[target_categorical_columns]

        numeric_features = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
            'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'swin', 'stcpb', 'dtcpb', 'dwin', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst'
        ]

        scaled_numeric_features = feature_df[numeric_features].apply(np.log1p)

        processed_features = np.hstack([scaled_numeric_features, categorical_encoded.values])

        return processed_features
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "Can't send Data."}), 400

        features = preprocess_features(data)

        prediction = model.predict(features)
        prediction_probabilities = [round(prob, 4) for prob in prediction[0].tolist()]

        class_labels = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal',
                        'Reconnaissance', 'Shellcode', 'Worms']
        
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_label = class_labels[predicted_class_index]

        response = {
            "probabilities": prediction_probabilities,
            "predicted_class": predicted_class_label
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An Error occured: " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
