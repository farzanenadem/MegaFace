# **Face Embedding Extraction with MagFace and ONNX**

This project provides a tool for extracting face embeddings using the **MagFace** model and converting it to the **ONNX** format.

---

## **Section 1: Download the Model Weights**

First, download the model weights from the following link:  
[Download Model Weights](https://drive.google.com/file/d/1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG/view)  

After downloading, place the weight file in the `weight` folder alongside other files and directories.

---

## **Section 2: Run the Program**

To execute the program, run the `main.py` file using the following format:

python main.py --image1_path <path_to_image1> --image2_path <path_to_image2> --arch <model_architecture> --batch_size <batch_size> --cpu_mode <true_or_false> --embedding_size <embedding_size> --workers <num_of_workers> --model_path <path_to_model_weights> --onnx_output_path <path_to_save_onnx>
