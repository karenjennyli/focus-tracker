import csv
import time
from pynput.keyboard import Listener, KeyCode
from datetime import datetime

current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# Name of the output file with the Unix epoch timestamp
output_file = f'labels/focus_data_{current_timestamp}.csv'

# Key used mark focus (in this case it is the char 'f')
focus_key = KeyCode(char='f')
distracted_key = KeyCode(char='d')
neutral_key = KeyCode(char='n')

# on_press is called when a key is pressed
# we are only keeping track of the timestamps where the focus is True
def on_press(key):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    timestamp_epoch = datetime.now().timestamp()
    if key == focus_key:
        focus_state = "Focused"
        # Append the timestamp and focus value to the CSV file
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, timestamp_epoch, focus_state])
            print(f"Focus marked at {timestamp} and epoch timestamp is {timestamp_epoch}")

    elif key == distracted_key:
        focus_state = "Distracted"
        # Append the timestamp and focus value to the CSV file
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, timestamp_epoch, focus_state])
            print(f"Distracted marked at {timestamp} and epoch timestamp is {timestamp_epoch}")
    
    elif key == neutral_key:
        focus_state = "Neutral"
        # Append the timestamp and focus value to the CSV file
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, timestamp_epoch, focus_state])
            print(f"Neutral marked at {timestamp} and epoch timestamp is {timestamp_epoch}")

    
def main():
    # Check if the CSV file exists and write headers if necessary
    try:
        with open(output_file, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'timestamp_epoch', 'focus_state'])
    except FileExistsError:
        pass  # File already exists, no need to add headers

    # Start listening for the focus key press
    with Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    print("Press 'f' to mark focus. Press 'Ctrl+C' to stop.")
    main()
