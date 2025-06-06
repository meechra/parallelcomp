import streamlit as st
import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

def convert_to_gray(image_bytes):
    """
    Convert image bytes to a grayscale image.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def sequential_conversion(image_bytes, repeats):
    """
    Run grayscale conversion sequentially for a given number of iterations.
    Returns the first grayscale image and the total execution time.
    """
    start_time = time.perf_counter()
    results = []
    for _ in range(repeats):
        gray = convert_to_gray(image_bytes)
        results.append(gray)
    end_time = time.perf_counter()
    return results[0], end_time - start_time

def parallel_conversion(image_bytes, repeats, max_workers):
    """
    Run grayscale conversion in parallel using ThreadPoolExecutor.
    Returns the first grayscale image and the total execution time.
    """
    start_time = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_to_gray, image_bytes) for _ in range(repeats)]
        for future in as_completed(futures):
            results.append(future.result())
    end_time = time.perf_counter()
    return results[0], end_time - start_time

def main():
    st.title("Sequential vs. Parallel Grayscale Conversion Comparison")
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        # Decode the original image for display
        original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        st.subheader("Original Image")
        st.image(original_image, channels="BGR", caption="Original Image", use_container_width=True)
        
        # Slider for selecting the number of iterations
        repeats = st.slider("Select the number of iterations for conversion:", 
                            min_value=5, max_value=50, value=20, step=5)
        
        # Slider for selecting the number of threads for parallel conversion
        max_threads = os.cpu_count() or 4  # Fallback in case os.cpu_count() returns None
        num_workers = st.slider("Select the number of parallel workers (threads):", 
                                min_value=1, max_value=max_threads, value=max_threads, step=1)
        
        st.write("Processing conversions...")
        
        # Sequential conversion
        seq_gray, seq_time = sequential_conversion(image_bytes, repeats)
        # Parallel conversion using the specified number of workers
        par_gray, par_time = parallel_conversion(image_bytes, repeats, num_workers)
        
        st.write(f"Sequential conversion ({repeats} iterations) took: {seq_time:.4f} seconds")
        st.write(f"Parallel conversion ({repeats} iterations) with {num_workers} worker(s) took: {par_time:.4f} seconds")
        
        # Calculate speedup and efficiency
        speedup = seq_time / par_time if par_time > 0 else float('inf')
        efficiency = speedup / num_workers if num_workers > 0 else float('inf')
        st.write(f"**Speedup Achieved:** {speedup:.2f}x")
        st.write(f"**Efficiency:** {efficiency:.2f} (Speedup per worker)")
        
        # Display the grayscale images side by side for comparison
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sequential Conversion Output")
            st.image(seq_gray, caption="Sequential", use_container_width=True)
        with col2:
            st.subheader("Parallel Conversion Output")
            st.image(par_gray, caption="Parallel", use_container_width=True)
        
        # Visualize the timing comparison with a bar chart
        df = pd.DataFrame({
            "Conversion": ["Sequential", "Parallel"],
            "Time (seconds)": [seq_time, par_time]
        })
        st.subheader("Conversion Time Comparison")
        st.bar_chart(df.set_index("Conversion"))

if __name__ == "__main__":
    main()
