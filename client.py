import requests
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="API URL")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Error: Image file not found.")
        return

    try:
        with open(args.image, "rb") as f:
            response = requests.post(args.url, files={"file": f})

        if response.status_code == 200:
            print("Response from server:")
            print(response.json())
        else:
            print(f"Server Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Connection Error: {e}")


if __name__ == "__main__":
    main()
