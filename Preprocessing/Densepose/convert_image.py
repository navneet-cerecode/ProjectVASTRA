import cv2
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--input", type=str, help="Path to input image", required=True)
parser.add_argument("--out", type=str, help="Path to save output image", required=True)
args = parser.parse_args()

# Logger and Predictor initialization
logger = GetLogger.logger(__name__)
predictor = Predictor()

# Read input image
image_path = args.input
image = cv2.imread(image_path)

if image is None:
    logger.error(f"Failed to load image: {image_path}")
    exit(1)

# Apply DensePose prediction
out_image, out_image_seg = predictor.predict(image)

# Save the output image
cv2.imwrite(args.out, out_image_seg)
logger.info(f"Saved output image to {args.out}")
