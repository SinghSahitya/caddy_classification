# CADDY - 3D CAD Model Classifier 

A deep learning application for classifying 3D CAD models using PointNet++ architecture. The system accepts STL, STEP, and OFF files, converts them to point clouds, and classifies them into common object categories. **[Link](https://superb-hamster-8a3759.netlify.app/)**


- 🔄 **Multi-format Support**: Upload CAD models in STL, STEP, and OFF formats
- 🔄 **Automatic Conversion**: Seamless conversion between CAD formats
- 🧠 **Deep Learning Classification**: Powered by PointNet++ architecture
- 🔍 **Interactive Visualization**: 3D point cloud viewer with rotation and zoom
- 📊 **Confidence Metrics**: Displays classification confidence scores

## 🏗️ Architecture

```
┌───────────────┐      ┌──────────────────────────┐      ┌───────────────┐
│    React      │──────▶    Node.js + Express    │──────▶   PyTorch     │
│   Frontend    │◀─────│     Backend API         │◀─────│   PointNet++   │
└───────────────┘      └──────────────────────────┘      └───────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  File Conversion   │
                         │  STL/STEP → OFF    │
                         └────────────────────┘
```

## 📝 Usage

1. Visit the application in your browser
2. Upload a CAD file using the drag-and-drop interface
3. Supported formats: `.off`, `.stl`, `.step`, `.stp`
4. View the classification results and 3D visualization

## 🧠 How It Works

### PointNet++ Architecture

PointNet++ processes point clouds directly through:

1. Hierarchical sampling and grouping
2. Multi-scale feature extraction
3. Feature aggregation
4. Classification

### File Conversion Pipeline

```
Upload CAD File → File Format Check → Format-Specific Conversion → Point Cloud Generation → PointNet++ Classification
```

## 📂 Project Structure

```
.
├── frontend/                # React frontend application
│   ├── public/              # Static files
│   └── src/                 # Source code
│       ├── components/      # React components
│       └── styles/          # CSS styles
│
├── backend/                 # Node.js backend
│   ├── uploads/             # Temporary storage for uploads
│   ├── python_scripts/      # Python scripts for ML and conversion
│   └── server.js            # Express server
```

---

Developed with ❤️ by [Sahitya Singh]
