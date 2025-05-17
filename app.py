import streamlit as st
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import folium_static, st_folium
import folium.plugins
import random
import requests
from io import BytesIO
import json

# Set page configuration with Light theme
st.set_page_config(
    page_title="Land Type Classification",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for Light theme
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
    }
    .stSelectbox label, .stFileUploader label {
        color: #2c3e50;
        font-weight: bold;
    }
    .uploadedFileData {
        color: #333333 !important;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .css-145kmo2 {
        color: #333333 !important;
    }
    .css-1offfwp p {
        color: #333333 !important;
    }
    div[data-testid="stMarkdownContainer"] > p {
        color: #333333;
    }
    .stMarkdown {
        color: #333333;
    }
    div.css-1kyxreq.etr89bj0 {
        color: #333333 !important;
    }
    .css-184tjsw p {
        color: #333333 !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Page title with Light theme icons
st.title("🛰️ Land Type Classification")
st.markdown("### Analyze Earth's surface features using satellite imagery")

# Define class names for the model
class_names = {
    0: "Annual Crop",
    1: "Forest",
    2: "Herbaceous Vegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "Permanent Crop",
    7: "Residential",
    8: "River",
    9: "Sea Lake"
}

def load_model_file():
    """Load the RGB model"""
    try:
        model_path = "model_v2.h5"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None

        # Custom loading method to handle compatibility issues
        try:
            # First attempt: standard loading
            model = load_model(model_path)
        except Exception as e:
            if 'batch_shape' in str(e):
                # Second attempt: load with custom_objects and compile=False
                st.info(f"Using compatibility mode to load model: {model_path}")
                
                # Clear any custom objects that might interfere
                tf.keras.utils.get_custom_objects().clear()
                
                # Load with compile=False to avoid batch_shape issues
                model = load_model(model_path, compile=False)
                
                # Manually compile the model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                # If it's a different error, re-raise it
                raise e
                
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(64, 64)):
    """Preprocess the image for RGB model"""
    # Resize image
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Process for RGB model
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] > 3:  # More than 3 channels
        img_array = img_array[:,:,:3]
    # Normalize
    img_array = img_array / 255.0
    
    # Expand dims for batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_analyze(image):
    """Make prediction and return results with analysis"""
    model = load_model_file()
    
    if model is None:
        return None
        
    # Preprocess image for the model
    processed_img = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx]) * 100
    
    return {
        "class_idx": class_idx,
        "class_name": class_names[class_idx],
        "confidence": confidence
    }

def display_analysis(result):
    """Display analysis based on prediction results with Plotly visualizations"""
    if result is None:
        return
        
    st.subheader("🔍 Analysis Results")
    
    # Create a confidence gauge with Plotly - Updated for light theme
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['confidence'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'color': '#2c3e50', 'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "#f5f5f5",
            'borderwidth': 2,
            'bordercolor': "#dddddd",
            'steps': [
                {'range': [0, 50], 'color': '#f2f2f2'},
                {'range': [50, 75], 'color': '#d9e6f2'},
                {'range': [75, 90], 'color': '#a6c9e2'},
                {'range': [90, 100], 'color': '#6baed6'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#ffffff",
        font={'color': "#333333", 'family': "Arial"},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display spectral bands information
    st.markdown("### Spectral Bands Information")
    
    st.markdown("""
    **RGB Model uses 3 spectral bands:**
    - **Red (0.6-0.7 μm)**: Sensitive to plant chlorophyll absorption
    - **Green (0.5-0.6 μm)**: Peak reflectance for vegetation
    - **Blue (0.4-0.5 μm)**: Good for distinguishing soil types and built environments
    """)
    
    # Add RGB visualization with Plotly - Updated for light theme
    bands_fig = px.bar(
        x=["Red", "Green", "Blue"],
        y=[0.65, 0.55, 0.45],  # Average wavelengths
        labels={"x": "Band", "y": "Wavelength (μm)"},
        color=["Red", "Green", "Blue"],
        color_discrete_map={"Red": "red", "Green": "green", "Blue": "blue"},
        title="RGB Spectral Bands"
    )
    bands_fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f5f5f5",
        font={'color': "#333333"},
    )
    
    # Generate a unique key for the chart
    chart_key = f"bands_fig_unique_{random.randint(0, 100000)}"
    st.plotly_chart(bands_fig, key=chart_key)
    
    # Display tips and insights based on land type
    st.markdown("### Tips and Insights")
    
    land_type_tips = {
        0: """
        **Annual Crop Land**
        - These areas show regular seasonal patterns in satellite imagery
        - Best monitored through time series analysis to track crop cycles
        - Consider exploring crop health monitoring with NDVI data
        - RGB models can detect different crop stages based on color changes
        """,
        
        1: """
        **Forest Area**
        - Forests are critical carbon sinks and biodiversity hotspots
        - Monitor for deforestation using time series analysis
        - RGB analysis can detect forest types and density based on color signatures
        - Consider conservation initiatives in these areas
        """,
        
        2: """
        **Herbaceous Vegetation**
        - These areas typically include grasslands, meadows, and shrublands
        - RGB analysis can detect vegetation density and types
        - Consider grazing potential or monitoring for invasive species
        - May provide ecosystem services like erosion control and wildlife habitat
        """,
        
        3: """
        **Highway/Road Infrastructure**
        - Built infrastructure detection is typically strongest in RGB models
        - Consider monitoring buffer zones alongside highways for environmental impact
        - Urban expansion typically follows transportation corridors
        - Linear features are distinctive in satellite imagery
        """,
        
        4: """
        **Industrial Area**
        - Industrial zones typically show distinctive textures and colors
        - Consider environmental monitoring for pollution effects
        - RGB analysis can detect building shapes, parking lots, and industrial features
        - Regular monitoring recommended for regulatory compliance
        """,
        
        5: """
        **Pasture Land**
        - Pastures show moderate green values with seasonal variations
        - Monitor for overgrazing using vegetation color changes
        - Consider rotational grazing strategies to optimize land use
        - RGB provides good differentiation from cropland
        """,
        
        6: """
        **Permanent Crop**
        - These areas (orchards, vineyards, etc.) show distinct patterns
        - Less seasonal variation than annual crops
        - RGB bands can detect row patterns and crop structure
        - Consider precision agriculture techniques for irrigation and fertilization
        """,
        
        7: """
        **Residential Area**
        - Mixed spectral signature including buildings and vegetation
        - Consider urban heat island effect monitoring
        - Opportunity for green space planning and development
        - RGB models typically perform well for urban classification
        """,
        
        8: """
        **River**
        - Water features have distinctive spectral signatures in all bands
        - Monitor for water quality changes and flooding
        - Consider riparian zone health assessment
        - RGB can detect water flow patterns and depth variations
        """,
        
        9: """
        **Sea/Lake**
        - Large water bodies are easily identifiable in all spectral bands
        - Monitor for algal blooms or pollution events
        - Consider shoreline change analysis over time
        - Water color variations can indicate depth and quality issues
        """
    }
    
    st.markdown(land_type_tips[result["class_idx"]])
    
    # Add a radar chart for land characteristic ratings - Updated for light theme
    characteristics = {
        0: {"Vegetation": 0.7, "Water": 0.3, "Urban": 0.1, "Soil": 0.8, "Seasonal Variation": 0.9},
        1: {"Vegetation": 0.9, "Water": 0.4, "Urban": 0.1, "Soil": 0.2, "Seasonal Variation": 0.5},
        2: {"Vegetation": 0.8, "Water": 0.3, "Urban": 0.1, "Soil": 0.5, "Seasonal Variation": 0.7},
        3: {"Vegetation": 0.2, "Water": 0.1, "Urban": 0.9, "Soil": 0.4, "Seasonal Variation": 0.1},
        4: {"Vegetation": 0.1, "Water": 0.3, "Urban": 0.9, "Soil": 0.5, "Seasonal Variation": 0.2},
        5: {"Vegetation": 0.8, "Water": 0.2, "Urban": 0.1, "Soil": 0.6, "Seasonal Variation": 0.5},
        6: {"Vegetation": 0.8, "Water": 0.4, "Urban": 0.2, "Soil": 0.5, "Seasonal Variation": 0.4},
        7: {"Vegetation": 0.5, "Water": 0.3, "Urban": 0.9, "Soil": 0.3, "Seasonal Variation": 0.3},
        8: {"Vegetation": 0.3, "Water": 0.9, "Urban": 0.1, "Soil": 0.2, "Seasonal Variation": 0.4},
        9: {"Vegetation": 0.1, "Water": 1.0, "Urban": 0.0, "Soil": 0.0, "Seasonal Variation": 0.3}
    }
    
    char_data = characteristics[result["class_idx"]]
    
    radar_fig = go.Figure()
    
    radar_fig.add_trace(go.Scatterpolar(
        r=list(char_data.values()),
        theta=list(char_data.keys()),
        fill='toself',
        name=result["class_name"],
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor="#f5f5f5"
        ),
        showlegend=False,
        title="Land Type Characteristics",
        paper_bgcolor="#ffffff",
        font={'color': "#333333"},
    )
    
    st.plotly_chart(radar_fig)

def capture_map_screenshot(bbox, width=400, height=400):
    """
    Capture a map screenshot (satellite imagery) from ArcGIS
    based on the bounding box.
    """
    south, west, north, east = bbox
    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "size": f"{width},{height}",
        "format": "jpg",
        "f": "image",
        "imageSR": "3857"
    }
    resp = requests.get(url, params=params)
    return Image.open(BytesIO(resp.content))

# Load shapes from JSON before creating the map and tabs
if "last_drawings.json" in os.listdir():
    with open("last_drawings.json", "r") as f:
        saved_shapes = json.load(f)
    if 'last_drawings' not in st.session_state or not st.session_state.last_drawings:
        st.session_state.last_drawings = saved_shapes

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Upload Image", "🌎 Map Selection"])

# Tab 1: Image Upload
with tab1:
    st.header("Upload Satellite Image")
    
    uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png", "tif"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        # Add CSS to center the image container
        st.markdown("""
            <style>
            .centered-image {
                display: flex;
                justify-content: center;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Use a single column with centered styling
        st.markdown('<div class="centered-image">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image" ,width=400)
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Make prediction and get analysis
        with st.spinner("Analyzing image..."):
            result = predict_and_analyze(image)
            
        if result:
            # Show detailed analysis
            
            # Display prediction with confidence
            st.markdown(f"### Predicted Land Type: **{result['class_name']}**")
            st.markdown(f"### Confidence: **{result['confidence']:.2f}%**")

            display_analysis(result)

# Tab 2: Map Selection
with tab2:
    st.header("Select Region on Map")
    
    # Instructions for map use
    st.markdown("""
    #### Instructions:
    1. Navigate to your area of interest on the map
    2. Use the draw tool (rectangle icon) in the upper left corner to select your region
    3. Click the "Analyze Selected Region" button to analyze the selected area directly
    """)
    
    # Initialize session state for coordinates and captured image if not exists
    if 'map_coords' not in st.session_state:
        st.session_state.map_coords = None
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    # Add a selectbox to choose between tile providers
    tile_options = ["OpenStreetMap", "Esri World Imagery"]
    selected_tile = st.selectbox("Select Map Style", tile_options)

    if selected_tile == "OpenStreetMap":
        tiles = "OpenStreetMap"
        attr = "OpenStreetMap"
    else:
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        attr = 'Esri World Imagery'

    # Initialize map centered on a default location with satellite imagery
    m = folium.Map(
        location=[30.0, 31.0], 
        zoom_start=5, 
        control_scale=True,
        tiles=tiles,
        attr=attr,
    )
    
    # On subsequent runs, add saved shapes to the map
    if 'last_drawings' in st.session_state and st.session_state.last_drawings:
        for shape in st.session_state.last_drawings:
            folium.features.GeoJson(shape['geometry']).add_to(m)
    
    # Add a layer control to allow switching between satellite and street map
    folium.LayerControl().add_to(m)

    # Add draw control to allow selecting rectangular areas and store last drawn coordinates
    draw_options = {
        'polyline': False,
        'polygon': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'rectangle': True,
    }
    
    draw = folium.plugins.Draw(
        draw_options=draw_options,
        edit_options={'edit': False},
        position='topleft'
    )
    draw.add_to(m)
    
    # Use st_folium to capture drawn shapes
    map_data = st_folium(m, width=800, height=500)

    # Check if any shapes were drawn
    if map_data and "all_drawings" in map_data:
        # Store drawings in session state
        st.session_state.last_drawings = map_data["all_drawings"]
        # Save the new shapes to a JSON file
        with open("last_drawings.json", "w") as f:
            json.dump(st.session_state.last_drawings, f)
        drawings = map_data["all_drawings"]
        if drawings:
            # Get the latest shape drawn (rectangle)
            shape = drawings[-1]
            # Extract bounding box from shape
            if "geometry" in shape and shape["geometry"]["type"] == "Polygon":
                coordinates = shape["geometry"]["coordinates"][0]
                # Folium returns corners in [lng, lat] order
                lats = [c[1] for c in coordinates]
                lngs = [c[0] for c in coordinates]
                # Compute bounding box
                south, north = min(lats), max(lats)
                west, east = min(lngs), max(lngs)
                st.session_state.map_coords = [south, west, north, east]
                st.session_state.captured_image = None  # Reset captured image

    # Display coordinates if available
    if st.session_state.map_coords:
        try:
            south, west, north, east = st.session_state.map_coords
            st.success(f"✅ Region selected: {north:.4f}°N, {west:.4f}°E to {south:.4f}°N, {east:.4f}°E")
            
            # Automatically fetch the bounding box image
            screenshot = capture_map_screenshot((south, west, north, east))
            st.image(screenshot, caption="Selected Region", width=400)
            
        except:
            st.warning("Region selected, but coordinates format is unexpected")
    
    # Add a button to trigger area capture and analysis
    analyze_button = st.button("Analyze Selected Region", key="analyze_region_btn")
    
    if analyze_button:
        if not st.session_state.map_coords:
            st.warning("⚠️ Please select a region on the map first by using the rectangle tool.")
        else:
            with st.spinner("Processing selected region..."):
                # Get live image from bounding box
                if st.session_state.captured_image is None:
                    picture = capture_map_screenshot(st.session_state.map_coords)
                    st.session_state.captured_image = picture
                else:
                    picture = st.session_state.captured_image
                
                # Run classification directly on live_image
                result = predict_and_analyze(picture)
                if result:
                    st.markdown(f"### Predicted Land Type: **{result['class_name']}**")
                    st.markdown(f"### Confidence: **{result['confidence']:.2f}%**")
                    display_analysis(result)

# Add sidebar information with light theme
st.sidebar.title("🛰️ About this App")
st.sidebar.info(
    "This satellite-based application uses deep learning models to classify land types from orbital imagery. "
    "You can either upload your own images or select regions from the map interface."
)

st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.markdown("""
**RGB Model:** Uses standard 3-band imagery for classification.

This model can identify different land types based on visible light patterns, textures, and colors captured in satellite imagery.
""")

st.sidebar.markdown("### 🌎 Dataset Information")
st.sidebar.markdown("""
The model was trained on satellite imagery containing the following land types:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea Lake
""")

# Footer
st.markdown("---")
st.markdown("🛰️ Developed for DEPI Land Type Classification Project • Satellite Imagery Analysis Platform • © 2025")
