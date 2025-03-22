#!/usr/bin/env python
import requests
import argparse
import os
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test the YOLO prediction API")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--url", default="http://localhost:5000/predict", help="API endpoint URL"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    try:
        img = Image.open(args.image)
        if args.verbose:
            print(f"Image opened successfully: {args.image}")
            print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    try:
        with open(args.image, "rb") as f:
            files = {"image": f}
            response = requests.post(args.url, files=files)
    except Exception as e:
        print(f"Error sending request: {e}")
        return

    print(f"Status Code: {response.status_code}")

    try:
        json_response = response.json()
        print(f"Response: {json_response}")

        # Additional info for successful predictions
        if (
            args.verbose
            and response.status_code == 200
            and json_response.get("success")
        ):
            predictions = json_response.get("predictions", [])
            print(f"\nFound {len(predictions)} predictions:")
            for i, pred in enumerate(predictions, 1):
                print(
                    f"  {i}. {pred.get('class')} (confidence: {pred.get('confidence'):.2f})"
                )
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {response.text}")


if __name__ == "__main__":
    main()
