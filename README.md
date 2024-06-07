**Project Summary:**

The CarTag project involved the development of a real-time license plate detection system, leveraging state-of-the-art machine learning techniques and user-friendly deployment tools. The following is a detailed account of the various stages of the project, from data preparation to model deployment.

**Data Preparation:**

A pre-annotated dataset comprising 433 images served as the foundation for this project. Each image was accompanied by XML files containing detailed annotations. These XML files were meticulously processed to extract the necessary information and convert the annotations into the YOLO format, which is a requirement for training the YOLOv8 model. This conversion was crucial to ensure the data was in the optimal format for the training process.

**Model Training:**

The core of the CarTag system is the YOLOv8 model, known for its efficiency and accuracy in object detection tasks. The model was trained over 100 epochs, a process that involved iterative optimization to enhance its performance. Throughout the training, various hyperparameters were fine-tuned to achieve the best possible results.

**Performance Metrics:**

Upon completion of the training, the model's performance was evaluated using standard metrics in object detection. The mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 0.5 was recorded at 0.7, indicating a high level of precision in detecting license plates. Furthermore, the model achieved a mAP of 0.5 at IoU=0.5:0.95, demonstrating robust performance across varying levels of IoU thresholds. These metrics underscore the model's effectiveness in identifying license plates with both high precision and consistency.
