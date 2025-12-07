# convert_model.py - FIXED for TFLite conversion with BatchNormalization
import tensorflow as tf
import numpy as np

print("Loading malaria CNN model...")
model = tf.keras.models.load_model("models/malaria_cnn.h5")

print("Model loaded successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# FIX: Use optimizations that handle batch normalization properly
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations (helps with batch normalization issues)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set experimental options to handle batch normalization
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = False

# Alternative: If above still fails, use representative dataset
# This ensures the converter can properly handle all operations
def representative_dataset():
    for _ in range(100):
        # Generate random samples matching your input shape (128x128x3)
        data = np.random.rand(1, 128, 128, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

print("\nConverting to TFLite format...")
try:
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open("models/malaria_cnn.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("✅ Model successfully converted to TFLite!")
    print(f"✅ Saved to: models/malaria_cnn.tflite")
    print(f"✅ File size: {len(tflite_model) / (1024*1024):.2f} MB")
    
    # Test the converted model
    print("\nTesting converted model...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details[0]['shape']}")
    print(f"Output details: {output_details[0]['shape']}")
    
    # Run a test inference
    test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Test inference successful! Output: {test_output[0][0]:.4f}")
    print("\n🚀 Ready for deployment to Render.com!")
    
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    print("\nTrying alternative method (quantization)...")
    
    # ALTERNATIVE METHOD: Use post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 128, 128, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
        tf.lite.OpsSet.SELECT_TF_OPS      # Enable TensorFlow ops (fallback)
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    with open("models/malaria_cnn.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("✅ Model converted with TF ops fallback!")
    print(f"✅ Saved to: models/malaria_cnn.tflite")
    print(f"✅ File size: {len(tflite_model) / (1024*1024):.2f} MB")