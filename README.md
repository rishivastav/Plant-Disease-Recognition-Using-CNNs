# Plant-Disease-Recognition-Using-CNNs
CNN‑based plant disease recognition using a Small CNN and SimCLR‑pretrained self supervised ResNet‑50 models. Trained on PlantVillage, this repository includes research code, model files, and a demo script allowing users to upload leaf images and receive disease predictions.



# Overview
This project demonstrates the effectiveness of Convolutional Neural Networks (CNNs) in classifying plant leaf diseases. A CNN is a type of neural network designed to understand images by detecting patterns such as edges, textures, and shapes. CNNs are widely used for image classification tasks, and in this project they are applied to identify plant diseases from leaf photos.
To evaluate performance, I compared two models, a self‑supervised SimCLR‑pretrained ResNet‑50 and a small handbuilt CNN. This comparison explores whether larger, pretrained CNN models outperform smaller handcrafted models in plant disease recognition.

#🌐Live Demo
A web‑based leaf‑upload scanner will be available soon. Users will be able to upload leaf images and choose between the Small CNN or SimCLR‑pretrained ResNet‑50 model to receive disease predictions, confidence scores, and probability charts.

#📂 Repository Structure
app.py — Streamlit web demo
predict.py — CLI prediction script
models/ — trained model weights (Small CNN + SimCLR‑ResNet‑50)
src/ — training, preprocessing, and utility scripts
requirements.txt — Python dependencies
README.md — project documentation

#🌱 Dataset
This project uses the PlantVillage dataset, containing ~54,000 labeled leaf images across 38 disease classes. The dataset includes multiple crops (tomato, apple, potato, grape, etc.) and both healthy and diseased leaves. Images are standardized and ideal for CNN‑based classification.

#🧠 Models
Small CNN
A lightweight convolutional neural network built from scratch.
Designed to test how a simple architecture performs on plant disease classification.
SimCLR‑pretrained ResNet‑50
A large self‑supervised model pretrained using SimCLR.
Fine‑tuned on PlantVillage to evaluate whether self‑supervised feature learning improves accuracy.
This comparison explores whether handcrafted CNNs can compete with modern pretrained architectures.

#📊 Results
Small CNN: 82% accuracy
SimCLR‑ResNet‑50: 90.45% accuracy
The SimCLR‑pretrained model greatly outperformed the Small CNN, showing the benefits of self‑supervised learning for image classification.

#🚀 Installation

git clone https://github.com/yourusername/Plant-Disease-Recognition-Using-CNNs
cd Plant-Disease-Recognition-Using-CNNs
pip install -r requirements.txt

#🧪 Usage

Run prediction script
___________________________________
python predict.py --image leaf.jpg
___________________________________

Run the website
___________________________________
streamlit run app.py
___________________________________

#🔮 Future Work
Deploy the web demo publicly
Add mobile support
Test on real farm images
Add more self‑supervised models (BYOL, MoCo, DINO)

#📝 Citation
If you use this project, please cite:
Reyansh (2026). Plant Disease Recognition Using CNNs.
