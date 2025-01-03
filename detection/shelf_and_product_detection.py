import os
import cv2

class ShelfandProductDetector:
    # def __init__(self, shelf_model_path, product_model_path, output_dir):
    #     # self.shelf_model_path = shelf_model_path
    #     # self.product_model_path = product_model_path
    #     # self.output_dir = output_dir
        
    def __init__(self, 
            shelf_model,             # The loaded YOLO model for shelf detection
            product_model,           # The loaded YOLO model for product detection
            base_output_dir: str     # The base directory where we’ll save everything
            ):
        self.shelf_model = shelf_model
        self.product_model = product_model
        self.base_output_dir = base_output_dir

    def shelf_detect_and_crop(self, input_images_dir: str, conf_threshold=0.8):
        """
        1. Detect shelves in the original images (input_images_dir).
        2. Crop out the shelves.
        3. Save the cropped shelf images in a subdirectory of base_output_dir.
        4. Return the path to that subdirectory (so we can feed it to the next step).
        """
        # Create a folder for the cropped shelf images
        shelf_output_dir = os.path.join(self.base_output_dir, "shelf_crops")
        os.makedirs(shelf_output_dir, exist_ok=True)

        # Perform shelf detection
        # results = self.shelf_model.predict(...) returns an iterable with .boxes, .path, etc.
        results = self.shelf_model.predict(
            source=input_images_dir,
            save=True,
            save_txt=True,
            conf=conf_threshold,
        )

        # Crop each detection and save
        self._crop_images(results, shelf_output_dir)
        return shelf_output_dir

    def product_detect_and_crop(self, shelf_images_dir: str, conf_threshold=0.8):
        """
        1. Detect products in the shelf-cropped images (shelf_images_dir).
        2. Crop out the products.
        3. Save the cropped product images in another subdirectory of base_output_dir.
        4. Return the path to that subdirectory.
        """
        product_output_dir = os.path.join(self.base_output_dir, "product_crops")
        os.makedirs(product_output_dir, exist_ok=True)

        # Perform product detection
        results = self.product_model.predict(
            source=shelf_images_dir,
            save=True,
            save_txt=True,
            conf=conf_threshold,
        )

        # Crop each detection and save
        self._crop_images(results, product_output_dir)
        return product_output_dir

    def detect_shelves_then_products(self, input_images_dir: str,
                                     shelf_conf=0.8,
                                     product_conf=0.6):
        """
        This is a convenience method that:
          1. Detects shelves and crops them.
          2. Detects products on those shelf-cropped images and crops them.
          3. Returns:
             - shelf_output_dir
             - product_output_dir
        """
        shelf_output_dir = self.shelf_detect_and_crop(input_images_dir, conf_threshold=shelf_conf)
        product_output_dir = self.product_detect_and_crop(shelf_output_dir, conf_threshold=product_conf)
        return shelf_output_dir, product_output_dir

    def _crop_images(self, results, output_dir):
        """
        A helper method to iterate over the detection results and crop out each bounding box.
        Saves images to `output_dir`.
        """
        for result in results:
            image_path = result.path  # Path to the original image
            image = cv2.imread(image_path)

            if image is None:
                print(f"[WARN] Failed to load image: {image_path}")
                continue

            if result.boxes is None:
                print(f"[INFO] No detections found for image: {image_path}")
                continue

            for i, box in enumerate(result.boxes.data):
                # box format: [x1, y1, x2, y2, conf, class_id]
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = box[4]
                class_id = int(box[5])

                # Crop
                cropped = image[y1:y2, x1:x2]

                # Build output path
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                out_fname = f"{base_name}_det_{i}_cls_{class_id}_conf_{confidence:.2f}.jpg"
                out_path = os.path.join(output_dir, out_fname)

                cv2.imwrite(out_path, cropped)

        print(f"[INFO] Cropping complete. Check folder: {output_dir}")


    # def crop_images(self, results):
    #      # Create output directory for cropped images
    #         # output_dir = "/home/azaa/detection_workspace/product_detection/Noodles_All_cropped"
    #         os.makedirs(self.output_dir, exist_ok=True)

    #         # Iterate over results
    #         for result in results:
    #             # Extract image information
    #             image_path = result.path  # Get the path to the original image
    #             image = cv2.imread(image_path)
                
    #             if image is None:
    #                 print(f"Failed to load image: {image_path}")
    #                 continue

    #             # Ensure boxes exist in the results
    #             if result.boxes is None:
    #                 print(f"No detections found for image: {image_path}")
    #                 continue

    #             # Iterate through the detections for the current image
    #             for i, box in enumerate(result.boxes.data):  # Use .data to access box details
    #                 # Extract bounding box coordinates
    #                 x1, y1, x2, y2 = map(int, box[:4])  # The first 4 values are the box coordinates
    #                 confidence = box[4]  # Confidence score
    #                 class_id = int(box[5])  # Class ID

    #                 # Crop the detected region
    #                 cropped_image = image[y1:y2, x1:x2]

    #                 # Save the cropped image
    #                 output_path = os.path.join(
    #                     self.output_dir, 
    #                     f"{os.path.basename(image_path).split('.')[0]}_product_{i}_class_{class_id}_conf_{confidence:.2f}.jpg"
    #                 )
    #                 cv2.imwrite(output_path, cropped_image)

    #         print("Cropping complete. Check the cropped images folder.")
        
        

    # def shelf_detect(self, images_path):

    #         # Run inference on new images
    #         results = self.shelf_model_path.predict(
    #             source=images_path,  # Path to the folder with new images
    #             save=True,                     # Save annotated images
    #             save_txt=True,                 # Save annotations in YOLO format (labels)
    #             conf=0.8,      # Confidence threshold (adjust as needed)
    #         )

    #         self.crop_images(results)
    
    
    # def product_detect(self, shelf_images_path):
    #         # Run inference on cropped shelf images
    #         results = self.shelf_model_path.predict(
    #             source=shelf_images_path,  # Path to the folder with new images
    #             save=True,                     # Save annotated images
    #             save_txt=True,                 # Save annotations in YOLO format (labels)
    #             conf=0.8,      # Confidence threshold (adjust as needed)
    #         )

    #         self.crop_images(results)
         
