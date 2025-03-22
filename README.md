## Toronto Visual AI Hackathon 2025

https://voxel51.com/computer-vision-events/visual-ai-hackathon-march-22-2025/

- [Toronto Visual AI Hackathon 2025](#toronto-visual-ai-hackathon-2025)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Development](#development)
- [Project Structure](#project-structure)
- [Team Slides](#team-slides)
- [References](#references)
- [Dataset(s) sources](#datasets-sources)

## Setup Instructions

### Prerequisites

- Python 3.11 (strict requirement)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/toronto-visual-ai-hackathon-2025.git
   cd toronto-visual-ai-hackathon-2025
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

   For development dependencies:
   ```bash
   poetry install --with dev
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Development

- Run code formatting and linting:
  ```bash
  poetry run ruff check --fix .
  ```

- Run tests:
  ```bash
  poetry run pytest
  ```

## Project Structure

```
toronto-visual-ai-hackathon-2025/
├── data/                # Dataset files (not tracked by git)
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
├── tests/               # Test files
├── pyproject.toml       # Project configuration
└── README.md            # This file
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

## Dataset(s) sources

- https://paperswithcode.com/
- https://www.perplexity.ai/
- https://huggingface.co/datasets
- Git dorking with `site:arxiv.org`
- community.voxel51.com discord [#datasets](https://discord.com/channels/1266527359511564372/1267989328164946042) channel
- GitHub
- https://docs.voxel51.com/dataset_zoo/index.html
- https://www.kaggle.com/datasets