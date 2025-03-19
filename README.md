---

# Image Caption Generator using Pretrained ResNet101 and LSTM

This repository contains a Jupyter Notebook that implements an image caption generator. The project uses a pretrained ResNet101 as the image encoder and an LSTM network as the decoder to generate captions for images from the Flickr8k dataset.

## Architecture Diagram

![Architecture Diagram](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/architecture.png)

*The architecture diagram was inspired by the article "[Solving an Image Captioning Task Using Deep Learning](https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/)" on Analytics Vidhya.*

## Project Structure

- **notebook.ipynb**: Main notebook containing code for data preprocessing, model building, training, and evaluation.
- **architecture_diagram.png**: Diagram illustrating the model architecture (pretrained ResNet101 + LSTM).
- **src/**: Directory to store sample output images with generated captions.

## Model Overview

- **Encoder (Pretrained ResNet101)**  
  The encoder uses a ResNet101 model (with the final fully connected layer removed) to extract high-level features from images. A new fully connected layer maps these features to a lower-dimensional embedding space.

- **Decoder (LSTM)**  
  The decoder is an LSTM network that takes the image features as the initial input and generates a caption one word at a time. It uses teacher forcing during training and a greedy search for inference.

## Usage

1. **Data Preparation**  
   - Download the Flickr8k dataset.
   - Ensure that the `captions.txt` file and the image files are placed in the correct directories.
   
2. **Training**  
   - Open the notebook and run all cells to preprocess the data, build the vocabulary, and train the model.
   - The notebook will display the training loss and sample generated captions as it trains.

3. **Inference**  
   - Use the trained model to generate captions on new images.
   - Sample output images along with their generated captions are saved in the `results/` directory.

## Sample Results

The table below shows example output images arranged in a two-column grid along with their corresponding captions. Replace the placeholder images and captions with your actual results.

| Result Image 1 | Result Image 2 |
| -------------- | -------------- |
| ![Result 1](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_01.png) | ![Result 2](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_02.png)|
| ![Result 3](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_03.png) | ![Result 4](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_04.png)|
| ![Result 1](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_05.png) | ![Result 2](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_06.png)|
| ![Result 3](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_07.png) | ![Result 4](https://github.com/janith99hansidu/Attention-CNN-LSTM-Image-Caption-Generator/blob/main/src/test_07.png)|

## Checkpoints

- Model checkpoints are saved as `checkpoint.pth` after training.
- To resume training or run inference, load the checkpoint using PyTorch’s `torch.load` method.

## References

- **ResNet101:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). This paper introduces the ResNet architecture, which is used as the image encoder in this project.
- **LSTM:** Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf). This paper explains the LSTM network architecture used for sequence prediction in caption generation.
- **Image Captioning:** Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). This work is one of the pioneering papers in the field of neural image captioning.
- **Flickr8k Dataset:** Hodosh, M., Young, P., & Hockenmaier, J. (2013). [Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics](https://www.cs.unc.edu/~mhodosh/captioning/).


## Acknowledgments

- This project is inspired by state-of-the-art image captioning research and leverages the capabilities of PyTorch and Torchvision.
- Thanks to the creators of the Flickr8k dataset and the open-source community for their contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---