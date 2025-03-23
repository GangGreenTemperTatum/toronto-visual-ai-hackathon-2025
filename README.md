## Toronto Visual AI Hackathon 2025

https://voxel51.com/computer-vision-events/visual-ai-hackathon-march-22-2025/

- [Toronto Visual AI Hackathon 2025](#toronto-visual-ai-hackathon-2025)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [Docker Deployment](#docker-deployment)
  - [Development](#development)
- [Project Structure](#project-structure)
- [API Usage](#api-usage)
- [Team Slides](#team-slides)
- [References](#references)
- [Dataset(s) sources](#datasets-sources)

<img src="./assets/toronto%20ai%20visual%20hackathon.jpg" alt="Demo screenshot 1" width="600">

<img src="./assets/toronto%20ai%20visual%20hackathon%202.jpg" alt="Demo screenshot 2" width="600">

## Setup Instructions

### Prerequisites

- Python 3.10+
- A trained YOLO model checkpoint

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GangGreenTemperTatum/toronto-visual-ai-hackathon-2025.git
   cd toronto-visual-ai-hackathon-2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

   For Poetry users:
   ```bash
   poetry install
   ```

3. **Important**: Update the model path in `src/toronto_visual_ai_hackathon/__init__.py`:
   ```python
   MODEL_PATH = "/path/to/your/model.pt"  # Change this to your model checkpoint
   ```

### Running the Application

You can run the application in two ways:

1. Using the `run.py` script:
   ```bash
   python run.py
   ```

2. Using the package directly:
   ```bash
   python -m toronto_visual_ai_hackathon
   ```

The API server will start at http://localhost:5000.

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t yolo-classifier .
   ```

2. Run the container with the models directory mounted:
   ```bash
   docker run -p 5000:5000 -v "$(pwd)/models:/app/models" yolo-classifier
   ```

   This ensures the model file is accessible inside the container.

### Development

- Run code formatting and linting:
  ```bash
  poetry run ruff check --fix .
  ```

- Run tests:
  ```bash
  # Option 1: Run directly
  python tests/test_request.py --image path/to/test/image.jpg

  # Option 2: Make executable and run
  chmod +x tests/test_request.py
  ./tests/test_request.py --image path/to/test/image.jpg

  # Option 3: Run as module (if package structure is set up)
  python -m tests.test_request --image path/to/test/image.jpg
  ```

## Project Structure

```
toronto-visual-ai-hackathon-2025/
├── data/                # Data files (not tracked by git)
├── models/              # Model checkpoints
│   └── best.pt          # Your trained YOLO model
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Utility scripts
├── src/                 # Source code
│   └── toronto_visual_ai_hackathon/
│       ├── __init__.py  # Main Flask application
│       └── __main__.py  # Entry point for running as module
├── tests/               # Test files
│   └── test_request.py  # Script to test the API
├── run.py               # Script to run the application
├── setup.py             # Package installation configuration
├── requirements.txt     # Dependencies
├── pyproject.toml       # Poetry configuration
├── Dockerfile           # Docker configuration
└── README.md            # This file
```

## API Usage

The application provides a single endpoint for image classification or detection:

**POST /predict**

Accepts form data with an 'image' file and returns classification or detection results, depending on the model type.

Example using curl:
```bash
# Option 1: Use absolute path
curl -X POST -F "image=@/Users/path/to/image" http://localhost:5000/predict

# Option 2: Use current directory path for a file in the same directory
curl -X POST -F "image=@./test_image.jpg" http://localhost:5000/predict

# Option 3: Expand tilde manually for Bash
curl -X POST -F "image=@$HOME/path/to/image" http://localhost:5000/predict
```

Example response for classification models:
```json
{
  "success": true,
  "model_type": "classify",
  "predictions": [
    {
      "class": "good_quality",
      "class_id": 0,
      "confidence": 0.95,
      "rank": 1
    },
    {
      "class": "poor_quality",
      "class_id": 1,
      "confidence": 0.05,
      "rank": 2
    }
  ]
}
```

Example response for detection models:
```json
{
  "success": true,
  "model_type": "detect",
  "predictions": [
    {
      "class": "face",
      "class_id": 0,
      "confidence": 0.95,
      "bbox": [10, 20, 100, 200]
    }
  ]
}
```

## Team Slides

- **Team**: `DeepFace-FiftyFive`
- Slides available [here](https://docs.google.com/presentation/d/1V_x1zA4pkNYdWTgE3Rv-pNXRsQe4Crmo5hNFl9GmnCI/edit?usp=sharing) TODO

## References

- [TMU Hackathon Intro slides](https://docs.google.com/presentation/d/1KIKnjJR1oDIHoTeX3S623QIkMwzJYDNHNRSl4doBrZQ/edit?slide=id.g2e929bf4542_0_719#slide=id.g2e929bf4542_0_719)
- https://voxelgpt.ai/lander
- https://docs.voxel51.com/user_guide/using_datasets.html
- https://github.com/jacobmarks/image-quality-issues
- https://github.com/swheaton/fiftyone-media-anonymization-plugin
- https://docs.voxel51.com/integrations/albumentations.html

## Dataset(s) sources

- Sample dataset used for hackathon [here](https://huggingface.co/datasets/GangGreenTemperTatum/lfw-sample-organized/tree/main)
- https://paperswithcode.com/
- https://www.perplexity.ai/
- https://huggingface.co/datasets
- Git dorking with `site:arxiv.org`
- community.voxel51.com discord [#datasets](https://discord.com/channels/1266527359511564372/1267989328164946042) channel
- GitHub
- https://docs.voxel51.com/dataset_zoo/index.html
- https://www.kaggle.com/datasets