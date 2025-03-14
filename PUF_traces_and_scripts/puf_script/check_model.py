from keras.models import load_model

# Load the trained model
model = load_model("../models/puf0/puf0.h5")

# Print model summary
model.summary()
