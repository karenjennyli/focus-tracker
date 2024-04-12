import time
import random
import datetime
import requests

# Define the function to send simulated EEG data
def send_simulated_eeg_data():
    flowCount = 0
    notInFlowCount = 0

    # Simulate sending data continuously
    while True:
        # Generate random EEG-like data
        data = {
            'pow': [random.uniform(0, 10) for _ in range(25)],  # Simulate power values
            # The timestamp of this sample. It is the number of seconds that have elapsed since 00:00:00 Thursday, 1 January 1970 UTC.
            'time': time.time()
        }

        # Process the data (simulated classification logic)
        input_features = data['pow'][0:5] + data['pow'][10:15] + data['pow'][20:25]
        # Simulated model inference (classification logic)
        predicted_class = 'Flow' if random.random() > 0.5 else 'Not in Flow'  # Simulate classification result

        print(f"The input vector is classified as: {predicted_class}")
        if predicted_class == 'Flow':
            flowCount += 1
        else:
            notInFlowCount += 1

        # Prepare data for sending
        formatted_time = datetime.datetime.fromtimestamp(data['time']).strftime('%H:%M:%S')
        data_to_send = {
            'timestamp_epoch': data['time'],
            'timestamp_formatted': formatted_time,
            'flow': predicted_class,
            'flowCount': flowCount,
            'notInFlowCount': notInFlowCount,
            'predictionSum': flowCount + notInFlowCount,
            'flowNotFlowRatio': flowCount / (flowCount + notInFlowCount) if flowCount + notInFlowCount > 0 else 0
        }

        # Send data to the server
        response = requests.post('http://127.0.0.1:8000/api/flow_data', json=data_to_send)
        if response.status_code == 201:
            print("EEG Flow State data successfully sent to Django")

        # Sleep to simulate 128Hz frequency (approximately 7.8 ms delay)
        time.sleep(5)  # 1 / 128 ≈ 0.0078 seconds

# Call the function to start sending simulated EEG data
send_simulated_eeg_data()
