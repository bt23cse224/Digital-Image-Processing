import cv2
import numpy as np

def hide_watermark(host_path, watermark_path, output_path):
    host_img = cv2.imread(host_path)
    watermark_img = cv2.imread(watermark_path)

    watermark_img = cv2.resize(watermark_img, (host_img.shape[1], host_img.shape[0]))
    assert host_img.shape == watermark_img.shape, "Images must be the same size and shape."

    watermark_bits = watermark_img >> 7
    host_img_cleared = host_img & 0b11111110
    watermarked_img = host_img_cleared | watermark_bits

    cv2.imwrite(output_path, watermarked_img)
    print(f"Watermark hidden successfully in '{output_path}'.")

def reveal_watermark(watermarked_path):
    watermarked_img = cv2.imread(watermarked_path)
    extracted_bits = watermarked_img & 0b00000001
    revealed_watermark = extracted_bits * 255

    # Show the revealed watermark
    cv2.imshow("Revealed Watermark", revealed_watermark)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

# === User Choice ===
mode = int(input("Enter 1 to 'hide' or 2 to 'reveal': "))

if mode == 1:
    host_path = input("Enter host image path: ").strip()
    watermark_path = input("Enter watermark image path: ").strip()
    output_path = input("Enter output image path: ").strip()
    hide_watermark(host_path, watermark_path, output_path)

elif mode == 2:
    watermarked_path = input("Enter watermarked image path: ").strip()
    reveal_watermark(watermarked_path)

else:
    print("Invalid mode. Please choose 'hide' or 'reveal'.")
    
