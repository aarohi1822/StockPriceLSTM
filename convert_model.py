from tensorflow.keras.models import load_model

print("Loading H5 model...")
model = load_model("models/lstm_model.h5", compile=False)

print("Saving as SavedModel...")
model.save("models/lstm_saved_model", save_format="tf")

print("DONE! SavedModel created at models/lstm_saved_model/")
