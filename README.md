## Toronto Visual AI Hackathon 2025

https://voxel51.com/computer-vision-events/visual-ai-hackathon-march-22-2025/

- [Toronto Visual AI Hackathon 2025](#toronto-visual-ai-hackathon-2025)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Development](#development)
- [Project Structure](#project-structure)
- [References](#references)

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

## References

- https://voxelgpt.ai/lander
- https://docs.voxel51.com/user_guide/using_datasets.html