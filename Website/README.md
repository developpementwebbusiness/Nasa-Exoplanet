# 🌌 S.T.A.R. Trackers — Exoplanet Detection Web Interface

> **NASA Space Apps Challenge 2025** — AI-Powered Exoplanet Classification System

A modern, interactive web application for analyzing Kepler telescope data and detecting exoplanets using advanced machine learning. Built with Next.js 15, React 19, and TypeScript, this interface connects to the STAR AI v2 Python backend for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Uploading Data](#1-uploading-data)
  - [Overview Mode](#2-overview-mode)
  - [Manual Classification](#3-manual-classification)
  - [Detailed Data View](#4-detailed-data-view)
  - [Model Management](#5-model-management)
- [Component Architecture](#component-architecture)
- [API Integration](#api-integration)
- [Data Format](#data-format)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🌟 Overview

The S.T.A.R. Trackers web interface is a comprehensive exoplanet detection system that combines:

- **AI-Powered Predictions**: Real-time classification using the STAR AI v2 neural network
- **Manual Classification**: Expert review and validation of AI predictions
- **Data Visualization**: Interactive charts and graphs for confidence analysis
- **Model Training**: Custom model creation from your own classifications
- **Batch Processing**: Handle thousands of exoplanet candidates efficiently

This application is designed for researchers, astronomers, and space enthusiasts who want to analyze Kepler telescope data and contribute to exoplanet discovery.

---

## ✨ Key Features

### 🚀 Data Upload & Processing

- **Multiple Format Support**: CSV, Excel (.xlsx/.xls), JSON, and TXT files
- **Intelligent Column Mapping**: Automatically maps your data columns to the required format
- **Batch Processing**: Handles large datasets (5000+ candidates) with automatic batching
- **Data Cleaning**: Removes comment lines and empty rows automatically

### 🧠 AI-Powered Analysis

- **Real-time Predictions**: Get instant AI classifications for all candidates
- **Confidence Scores**: View probability scores (0-100%) for each prediction
- **Fallback Mode**: Works offline with mock predictions if API is unavailable
- **Recheck Functionality**: Re-analyze data after model updates

### 📊 Interactive Visualization

- **Probability Graph**: Visual representation of AI confidence scores
- **Color-coded Classifications**: Easy identification of exoplanets, false positives, and uncertain cases
- **Detailed Data Table**: Sortable, searchable table with all candidate information
- **Candidate Modals**: Deep-dive view for individual exoplanet candidates

### 👨‍🔬 Manual Classification

- **Expert Review**: Classify candidates as exoplanet, not exoplanet, or unsure
- **Comment System**: Add detailed notes and observations for each candidate
- **Progress Tracking**: Visual indicators show classification progress
- **Navigation**: Easy browsing through candidates with keyboard shortcuts

### 🎯 Model Management

- **Custom Training**: Train new AI models using your classifications
- **Training Datasets**: Upload pre-labeled datasets for model training
- **Model Export/Import**: Download and share trained models
- **Multiple Models**: Manage and switch between different AI models
- **Version Control**: Track model versions and performance metrics

### 🎨 Modern UI/UX

- **Dark/Light Mode**: System-aware theme with manual toggle
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Smooth Animations**: Polished interactions with Framer Motion
- **Accessibility**: ARIA labels and keyboard navigation support

---

## 🛠 Technology Stack

### Frontend Framework

- **Next.js 15.5.4**: React framework with App Router and Turbopack
- **React 19.0.0**: Latest React with concurrent features
- **TypeScript 5**: Type-safe development

### UI Components

- **Radix UI**: Accessible, unstyled component primitives
- **Tailwind CSS 4.1.9**: Utility-first CSS framework
- **Framer Motion 11.15**: Animation library
- **Lucide React**: Beautiful icon library
- **Recharts 2.15**: Chart and data visualization library

### Data Processing

- **PapaParse 5.4.1**: CSV parsing and processing
- **Next Themes 0.4.4**: Theme management (dark/light mode)

### Development Tools

- **PostCSS**: CSS processing
- **ESLint**: Code linting
- **TypeScript ESLint**: TypeScript-specific linting

---

## 📦 Prerequisites

Before running the web interface, ensure you have:

1. **Node.js**: Version 20.x or higher
2. **Package Manager**: npm, yarn, pnpm, or bun
3. **Python Backend**: The STAR AI API server must be running (see `Python/server/README.md`)
4. **Modern Browser**: Chrome, Firefox, Edge, or Safari (latest versions)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/developpementwebbusiness/Nasa-Exoplanet.git
cd Nasa-Exoplanet/Website
```

### 2. Install Dependencies

Choose your preferred package manager:

```bash
# Using npm
npm install

# Using yarn
yarn install

# Using pnpm
pnpm install

# Using bun
bun install
```

### 3. Configure Environment Variables

Create a `.env.local` file in the `Website` directory:

```bash
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

**Optional Variables:**

- `NEXT_PUBLIC_API_BASE_URL`: Python backend URL (default: `http://localhost:8000`)

### 4. Start the Python Backend

Before starting the web interface, ensure the Python API server is running:

```bash
cd ../Python/server
python app.py
```

The API should be accessible at `http://localhost:8000`

### 5. Start the Development Server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

The application will be available at:

- **Local**: [http://localhost:3000](http://localhost:3000)
- **Network**: Check your terminal for the network address

---

## 📖 Usage Guide

### 1. Uploading Data

#### Supported File Formats

- **CSV** (`.csv`): Comma-separated values
- **Excel** (`.xlsx`, `.xls`): Microsoft Excel spreadsheets
- **JSON** (`.json`): JavaScript Object Notation
- **TXT** (`.txt`): Plain text with comma-separated values

#### Upload Methods

1. **Drag & Drop**: Drag your data file into the upload area
2. **File Browser**: Click "Select Data File" to choose from your computer

#### Column Mapping

After upload, you'll see the **Column Mapper** interface:

1. **Review Your Columns**: See all column names from your file
2. **Map to Standard Names**: Match your columns to the required 35 features
3. **Auto-Detection**: The system suggests mappings based on column names
4. **Confirm or Cancel**: Apply mappings or cancel to re-upload

**Required Features (35 total):**

```
OrbitalPeriod, OPup, OPdown, TransEpoch, TEup, TEdown,
Impact, ImpactUp, ImpactDown, TransitDur, DurUp, DurDown,
TransitDepth, DepthUp, DepthDown, PlanetRadius, RadiusUp, RadiusDown,
EquilibriumTemp, InsolationFlux, InsolationUp, InsolationDown,
TransitSNR, StellarEffTemp, SteffUp, SteffDown, StellarLogG,
LogGUp, LogGDown, StellarRadius, SradUp, SradDown, RA, Dec, KeplerMag
```

#### Processing

- **AI Analysis**: Automatic classification begins immediately after upload
- **Batch Processing**: Large datasets (5000+ candidates) are processed in batches
- **Progress Indicator**: Visual feedback during processing
- **Error Handling**: If API fails, fallback mode provides mock predictions

---

### 2. Overview Mode

After data upload, you'll see the **Overview** tab with comprehensive statistics and visualizations.

#### Statistics Panel

Displays real-time metrics:

- **Total Candidates**: Number of loaded exoplanet candidates
- **Exoplanets**: Confirmed exoplanets (manually classified)
- **Not Exoplanets**: False positives (manually classified)
- **Unsure**: Candidates requiring further review
- **Average Confidence**: Mean AI confidence score across all predictions

#### Probability Graph

Interactive confidence visualization:

- **X-axis**: Candidate index (numbered 1, 2, 3, ...)
- **Y-axis**: AI confidence score (0-100%)
- **Color Coding**:
  - 🟢 **Green**: Exoplanet (>70% confidence or manually classified)
  - 🔴 **Red**: Not an exoplanet (<30% confidence or manually classified)
  - 🟡 **Yellow**: Unsure (30-70% confidence)
  - 🔵 **Blue**: Unclassified

**Interactions:**

- **Hover**: View detailed information for each candidate
- **Click**: Select a candidate to view in the classification panel
- **Double-Click**: Open detailed modal for the candidate

---

### 3. Manual Classification

#### Classification Panel Features

**Navigation:**

- **Previous/Next Buttons**: Move through candidates sequentially
- **Jump to Index**: Type a candidate number to jump directly
- **Keyboard Shortcuts**: Use arrow keys for quick navigation

**Candidate Information:**
Displays key data for the selected candidate:

- **Name**: Candidate identifier (KOI name, Kepler name, or auto-generated)
- **AI Prediction**: Model's classification (Exoplanet / False Positive)
- **Confidence Score**: AI confidence percentage
- **Key Parameters**: Orbital period, radius, temperature, etc.

**Classification Actions:**
Three classification options:

1. **✓ Exoplanet** (Green): Confirmed exoplanet
2. **✗ Not Exoplanet** (Red): False positive
3. **? Unsure** (Yellow): Requires further analysis

**Comment System:**

- Add detailed observations and notes
- Markdown support for rich text formatting
- Comments saved automatically with classification
- Visible in exported data and training datasets

**Progress Tracking:**

- Visual indicator shows classification progress
- Percentage completed
- Count of classified vs. total candidates

---

### 4. Detailed Data View

Switch to the **Detailed Data** tab for comprehensive table view.

#### Data Table Features

**Column Display:**

- All 35 input features
- AI prediction and confidence score
- Manual classification (if assigned)
- Color-coded status indicators

**Sorting:**

- Click column headers to sort
- Multi-level sorting support
- Sort by confidence, name, any parameter

**Filtering:**

- Search across all columns
- Filter by classification status
- Filter by confidence ranges

**Row Actions:**

- Click any row to select the candidate
- Double-click to open detailed modal
- Keyboard navigation (arrow keys)

**Export Options:**

- Download filtered/sorted data as CSV
- Export with or without classifications
- Include comments in export

---

### 5. Model Management

#### Training New Models

**Requirements:**

- **Option 1**: Manually classify at least 10 candidates in the app
- **Option 2**: Upload a pre-labeled training dataset (CSV format)

**Training Steps:**

1. **Choose Training Data:**

   - Use your manual classifications from the app, OR
   - Upload a CSV file with pre-labeled training data

2. **Enter Model Name:**

   - Use lowercase letters, numbers, and underscores only
   - Example: `my_exoplanet_model` or `kepler_2025_v1`
   - Name must be unique

3. **Start Training:**

   - Click "Train Model" button
   - Training runs on the Python backend
   - Wait for completion (typically 30-60 seconds)

4. **View Results:**
   - Training accuracy displayed upon completion
   - Model saved automatically
   - Available for immediate use

**Training Dataset Format:**
If uploading a training CSV, it must include:

- All 35 required features (columns)
- A `label` column (1 = exoplanet, 0 = not exoplanet)
- Optional `comment` column for notes

Example training CSV structure:

```csv
OrbitalPeriod,OPup,OPdown,...,KeplerMag,label,comment
2.7,0.0,0.0,...,15.4,1,"Confirmed by spectroscopy"
5.4,0.1,0.1,...,16.2,0,"False positive - stellar activity"
```

#### Model Library

**Viewing Models:**

- All trained models listed with metadata
- Default model marked with a star ⭐
- Model version and type displayed
- Training sample count shown

**Model Actions:**

- **Download**: Export model as ZIP file (includes weights, scaler, encoder)
- **Import**: Upload previously exported models
- **Refresh**: Update the model list from the server

**Model Files:**
Each exported model ZIP contains:

```
model_name.zip
├── model_weights.pkl      # Neural network weights
├── scaler.pkl            # Data normalization scaler
└── label_encoder.pkl     # Label encoder
```

**Using Custom Models:**

1. Train a model in the app
2. Download the model ZIP file
3. Share with colleagues or use on another machine
4. Import via the "Import Model" button

---

## 🏗 Component Architecture

### Main Application (`app/page.tsx`)

Central component that orchestrates all functionality:

- State management for CSV data, predictions, and classifications
- API communication
- Component coordination

### Core Components

#### `CSVUploader`

Handles all file upload and processing:

- Drag-and-drop interface
- Multiple format parsing (CSV, Excel, JSON, TXT)
- Comment line removal
- Column mapper integration

#### `ColumnMapper`

Smart column mapping interface:

- Displays uploaded column names
- Suggests standard name mappings
- Validates required features
- Applies transformations

#### `ProbabilityGraph`

Interactive confidence visualization:

- Recharts-based scatter plot
- Color-coded by classification
- Click and hover interactions
- Responsive design

#### `ClassificationPanel`

Manual classification interface:

- Candidate navigation
- Classification buttons
- Comment system
- Progress tracking

#### `DataTable`

Comprehensive data grid:

- All candidate data display
- Sorting and filtering
- Row selection
- Export functionality

#### `CandidateModal`

Detailed candidate information:

- Full parameter display
- AI prediction details
- Classification history
- Quick classification actions

#### `ModelManager`

AI model training and management:

- Training interface
- Model library
- Export/import functionality
- Progress indicators

#### `StatsOverview`

Real-time statistics display:

- Total candidates
- Classification counts
- Average confidence
- Animated counters

#### `ThemeToggle`

Theme management:

- Dark/light mode switch
- System preference detection
- Persistent storage

---

## 🔌 API Integration

### Client Library (`lib/api-client.ts`)

The application uses a type-safe API client for communication with the Python backend.

#### Key Functions

**`predict(data, userId)`**

- Sends candidate data for AI classification
- Handles batch processing for large datasets
- Automatic retry on failure
- Returns predictions with scores and labels

**`csvRowToExoplanetData(row, index)`**

- Converts CSV rows to API format
- Maps various column name formats
- Fills missing values with defaults
- Ensures exactly 35 features

**`checkApiHealth()`**

- Verifies backend availability
- 5-second timeout
- Used for fallback mode detection

**`getModelInfo()`**

- Retrieves model metadata
- Feature count and names
- Model architecture details

**`downloadModel(modelId)`**

- Exports trained models
- Returns ZIP blob
- Supports specific or all models

#### Error Handling

The client implements robust error handling:

- **Network Errors**: Automatic fallback to mock predictions
- **Timeouts**: 5-second timeout on health checks
- **Invalid Data**: Validation before sending
- **Server Errors**: User-friendly error messages

#### Batch Processing

For large datasets (>5000 candidates):

1. Splits data into batches of 5000
2. Sends batches sequentially
3. Combines results
4. Provides progress feedback

---

## 📊 Data Format

### Input Data Requirements

#### Column Names

The application accepts multiple naming conventions:

- **Standard names**: `OrbitalPeriod`, `TransitDepth`, etc.
- **KOI format**: `koi_period`, `koi_depth`, etc.
- **Custom names**: Mapped via the Column Mapper

#### Feature Types

- **Numeric**: All 35 features must be numeric
- **Missing Values**: Replaced with 0.0
- **Invalid Values**: NaN and Infinity cause errors

#### Example Input (CSV)

```csv
name,koi_period,koi_period_err1,koi_period_err2,koi_time0bk,...
KOI-123,2.7,0.0,0.0,170.7,...
KOI-456,5.4,0.1,-0.1,180.2,...
```

### Output Data Format

#### Prediction Results

Each prediction includes:

```typescript
{
  name: string; // Candidate identifier
  score: number; // Confidence (0.0 to 1.0)
  label: boolean; // true = exoplanet, false = not
}
```

#### Classifications

Manual classifications stored as:

```typescript
{
  type: "exoplanet" | "not_exoplanet" | "unsure";
  comment?: string;     // Optional notes
}
```

---

## 💻 Development

### Project Structure

```
Website/
├── app/
│   ├── layout.tsx          # Root layout with providers
│   ├── page.tsx            # Main application page
│   ├── globals.css         # Global styles
│   └── favicon.ico         # App icon
├── components/
│   ├── csv-uploader.tsx    # File upload component
│   ├── column-mapper.tsx   # Column mapping interface
│   ├── probability-graph.tsx   # Confidence chart
│   ├── classification-panel.tsx # Manual classification
│   ├── data-table.tsx      # Detailed data grid
│   ├── candidate-modal.tsx # Candidate details modal
│   ├── model-manager.tsx   # AI model management
│   ├── stats-overview.tsx  # Statistics display
│   ├── theme-toggle.tsx    # Dark/light mode toggle
│   └── ui/                 # Reusable UI components
│       ├── button.tsx
│       ├── card.tsx
│       ├── input.tsx
│       └── ...
├── lib/
│   ├── api-client.ts       # Python backend API client
│   └── utils.ts            # Utility functions
├── public/                 # Static assets
├── package.json            # Dependencies and scripts
├── tsconfig.json           # TypeScript configuration
├── next.config.ts          # Next.js configuration
└── README.md               # This file
```

### Available Scripts

```bash
# Development server with hot reload
npm run dev

# Production build
npm run build

# Start production server
npm run start

# Run ESLint
npm run lint
```

### Development Server

The development server includes:

- **Hot Module Replacement**: Instant updates without page refresh
- **Fast Refresh**: Preserves component state during edits
- **Turbopack**: Next-generation bundler for faster builds
- **Source Maps**: Debug with original TypeScript code

### Building for Production

```bash
# Create optimized production build
npm run build

# Test production build locally
npm run start
```

Production builds include:

- Minification and compression
- Tree-shaking (removes unused code)
- Image optimization
- Code splitting
- Static generation for faster loads

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Connection Failed

**Symptoms:**

- "API unavailable" message
- Mock predictions instead of real AI predictions
- Connection timeout errors

**Solutions:**

- Ensure Python backend is running: `cd Python/server && python app.py`
- Check API URL in `.env.local`: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
- Verify backend is accessible: Visit `http://localhost:8000/docs`
- Check firewall settings

#### 2. CSV Upload Fails

**Symptoms:**

- File not recognized
- Parsing errors
- Wrong number of features

**Solutions:**

- Ensure file format is supported (CSV, Excel, JSON, TXT)
- Check for correct encoding (UTF-8 preferred)
- Remove special characters from column names
- Use Column Mapper to match your column names
- Verify numeric data (no text in number columns)

#### 3. Model Training Fails

**Symptoms:**

- Training never completes
- "Insufficient data" error
- Server error during training

**Solutions:**

- Classify at least 10 candidates manually
- Or upload a valid training dataset CSV
- Ensure Python backend has write permissions
- Check backend logs for detailed errors
- Verify model name is valid (lowercase, no spaces)

#### 4. Slow Performance

**Symptoms:**

- Laggy UI interactions
- Slow graph rendering
- Long prediction times

**Solutions:**

- Use batch processing for large datasets (>5000 rows)
- Close unnecessary browser tabs
- Clear browser cache
- Restart the development server
- Check system resources (CPU, memory)

#### 5. Theme Toggle Not Working

**Symptoms:**

- Dark mode not applying
- Theme resets on page reload

**Solutions:**

- Clear browser local storage
- Check browser console for errors
- Ensure cookies are enabled
- Try different browser

### Debug Mode

Enable debug logging:

```typescript
// In app/page.tsx, set:
const DEBUG_MODE = true;

// Check browser console for detailed logs
```

### Getting Help

If you encounter issues:

1. **Check the Console**: Open browser DevTools (F12) and look for errors
2. **Review Backend Logs**: Check Python server terminal output
3. **Verify Data Format**: Ensure your CSV matches expected format
4. **Test with Sample Data**: Use provided sample datasets
5. **Create an Issue**: Report bugs on GitHub with detailed information

---

## 🤝 Contributing

We welcome contributions from the community!

### How to Contribute

1. **Fork the Repository**

   ```bash
   git fork https://github.com/developpementwebbusiness/Nasa-Exoplanet.git
   ```

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**

   - Follow existing code style
   - Add comments for complex logic
   - Update documentation

4. **Test Your Changes**

   ```bash
   npm run lint
   npm run build
   ```

5. **Commit with Descriptive Messages**

   ```bash
   git commit -m "Add: New feature for batch export"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Describe your changes
   - Link related issues
   - Add screenshots if UI changes

### Code Style

- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Keep components small and focused
- Add JSDoc comments for functions
- Use meaningful variable names

### Testing

Before submitting:

- Test with different data formats
- Verify API integration works
- Check responsive design (mobile, tablet, desktop)
- Test dark and light modes
- Ensure accessibility (keyboard navigation)

---

## 📄 License

This project is part of the NASA Space Apps Challenge 2025.

**Open Source** — Free to use, modify, and distribute.

See the main repository for full license details.

---

## 🌟 Acknowledgments

- **NASA**: For Kepler mission data and inspiration
- **Space Apps Challenge**: For organizing this global event
- **Open Source Community**: For the amazing tools and libraries
- **Contributors**: Everyone who helped build and improve this project

---

## 📧 Contact

- **GitHub**: [developpementwebbusiness/Nasa-Exoplanet](https://github.com/developpementwebbusiness/Nasa-Exoplanet)
- **Space Apps**: [Official Website](https://www.spaceappschallenge.org/)

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and navigate
git clone https://github.com/developpementwebbusiness/Nasa-Exoplanet.git
cd Nasa-Exoplanet/Website

# 2. Install dependencies
npm install

# 3. Start Python backend (separate terminal)
cd ../Python/server && python app.py

# 4. Start web interface
npm run dev

# 5. Open browser
# Visit http://localhost:3000
```

Happy exoplanet hunting! 🌠
