import streamlit as st
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# Function to convert image bytes to grayscale
def convert_to_gray(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Sequential conversion: loop through the conversion repeatedly
def sequential_conversion(image_bytes, repeats=20):
    start_time = time.perf_counter()
    results = []
    for _ in range(repeats):
        gray = convert_to_gray(image_bytes)
        results.append(gray)
    end_time = time.perf_counter()
    # Return the first result and the elapsed time
    return results[0], end_time - start_time

# Parallel conversion: use ProcessPoolExecutor to convert concurrently
def parallel_conversion(image_bytes, repeats=20):
    start_time = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_to_gray, image_bytes) for _ in range(repeats)]
        results = [future.result() for future in futures]
    end_time = time.perf_counter()
    return results[0], end_time - start_time

# Streamlit interface
st.title("Sequential vs. Parallel Grayscale Conversion Comparison")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    
    # Show original image
    st.subheader("Original Image")
    original = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    st.image(original, channels="BGR", caption="Original Image", use_column_width=True)
    
    # Perform sequential conversion and measure time
    seq_gray, seq_time = sequential_conversion(image_bytes, repeats=20)
    st.write(f"Sequential conversion (20 iterations) took: {seq_time:.4f} seconds")
    
    # Perform parallel conversion and measure time
    par_gray, par_time = parallel_conversion(image_bytes, repeats=20)
    st.write(f"Parallel conversion (20 iterations) took: {par_time:.4f} seconds")
    
    # Display the grayscale images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sequential Conversion")
        st.image(seq_gray, caption="Sequential", use_column_width=True)
    with col2:
        st.subheader("Parallel Conversion")
        st.image(par_gray, caption="Parallel", use_column_width=True)
