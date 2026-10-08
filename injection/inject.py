import os
import piexif
from PIL import Image

def add_gps_to_image(input_path, output_path, lat, lon):
    try:
        # Load existing exif data safely
        exif_dict = piexif.load(input_path)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}}

    def deg_to_dms_rational(deg_float):
        deg = int(deg_float)
        min_float = (deg_float - deg) * 60
        minute = int(min_float)
        sec_float = (min_float - minute) * 60
        second = int(sec_float * 10000)
        return ((deg, 1), (minute, 1), (second, 10000))

    lat_ref = 'N' if lat >= 0 else 'S'
    lon_ref = 'E' if lon >= 0 else 'W'
    
    lat_dms = deg_to_dms_rational(abs(lat))
    lon_dms = deg_to_dms_rational(abs(lon))

    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode('utf-8')
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = lat_dms
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode('utf-8')
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = lon_dms

    exif_bytes = piexif.dump(exif_dict)
    
    # Open and save while explicitly maintaining the clean RGB/Grayscale structure
    with Image.open(input_path) as img:
        # If it's a palette or RGBA image, convert to standard RGB to prevent tensor conversion errors
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(output_path, "JPEG", exif=exif_bytes, quality=95)
        
    print(f"Success! Clean GPS-injected image created at: {output_path}")

if __name__ == "__main__":
    base_dir = "/Users/chitrabhanuhazra/Downloads/ CH CODING/CYCLON PREDICTION/injection"
    source_image = os.path.join(base_dir, "storm.jpg")
    output_image = os.path.join(base_dir, "odisha_storm_gps.jpg")
    
    target_lat = 19.8
    target_lon = 85.8
    
    add_gps_to_image(source_image, output_image, target_lat, target_lon)