import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import time

model = tf.keras.models.load_model("voice_robot_model.h5")

LABELS = ["go", "left", "right", "stop"]

FS = 16000
DURATION = 1

def record_audio():
    print("🎤 Listening...")

    audio = sd.rec(int(FS * DURATION),
                   samplerate=FS,
                   channels=1)

    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=FS, n_mfcc=40)

    # pad / truncate to fixed size
    if mfcc.shape[1] < 40:
        pad = 40 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)))
    else:
        mfcc = mfcc[:, :40]

    return mfcc

def predict_command():
    audio = record_audio()

    mfcc = extract_mfcc(audio)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc, verbose=0)

    index = np.argmax(pred)
    confidence = np.max(pred)

    label = LABELS[index]

    print(f"👉 Prediction: {label} | Confidence: {confidence:.2f}")

    return label, confidence

def main():
    print("🚀 Voice Robot Started")

    while True:
        label, conf = predict_command()

        # confidence treshold has to be at least 75% for the action to be executed
        if conf > 0.75:
            if label == "go":
                print("MOVE FORWARD")
            elif label == "left":
                print("TURN LEFT")
            elif label == "right":
                print("TURN RIGHT")
            elif label == "stop":
                print("STOP")

        else:
            print("⚠️ Low confidence - ignoring")

        # updates every 500 ms
        time.sleep(0.5)


if __name__ == "__main__":
    main()