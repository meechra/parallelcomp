import streamlit as st
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def convert_to_gray(image_bytes):
    """
    Convert image bytes to a grayscale image.
    """
    # Convert bytes to a NumPy array and decode it
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def sequential_conversion(image_bytes, repeats=20):
    """
    Process the image sequentially for a given number of iterations.
    Returns the first grayscale result and the total execution time.
    """
    start_time = time.perf_counter()
    results = []
    for _ in range(repeats):
        gray = convert_to_gray(image_bytes)
        results.append(gray)
    end_time = time.perf_counter()
    return results[0], end_time - start_time

def parallel_conversion(image_bytes, repeats=20):
    """
    Process the image in parallel using ProcessPoolExecutor for a given number of iterations.
    Returns the first grayscale result and the total execution time.
    """
    start_time = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_to_gray, image_bytes) for _ in range(repeats)]
        results = [future.result() for future in futures]
    end_time = time.perf_counter()
    return results[0], end_time - start_time

def main():
    st.title("Sequential vs. Parallel Grayscale Conversion Comparison")
    
    # Image upload widget
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        
        # Decode and display the original image
        original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        st.subheader("Original Image")
        st.image(original_image, channels="BGR", caption="Original Image", use_container_width=True)
        
        # Run sequential conversion and display the timing
        seq_gray, seq_time = sequential_conversion(image_bytes, repeats=20)
        st.write(f"Sequential conversion (20 iterations) took: {seq_time:.4f} seconds")
        
        # Run parallel conversion and display the timing
        par_gray, par_time = parallel_conversion(image_bytes, repeats=20)
        st.write(f"Parallel conversion (20 iterations) took: {par_time:.4f} seconds")
        
        # Display the grayscale images side by side for comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sequential Conversion")
            st.image(seq_gray, caption="Sequential", use_container_width=True)
        with col2:
            st.subheader("Parallel Conversion")
            st.image(par_gray, caption="Parallel", use_container_width=True)

if __name__ == '__main__':
    main()
