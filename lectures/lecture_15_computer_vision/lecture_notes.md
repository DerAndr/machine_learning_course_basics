# Lecture 15 Notes: Computer Vision

These notes summarize the concepts used in the practical notebook.

## Images as Tensors

Digital images are arrays of numbers:

- grayscale image: `height x width`;
- RGB image: `height x width x 3`;
- mini-batch for a neural network: usually `batch x channels x height x width`.

The same object can look very different after resizing, cropping, lighting changes, or color shifts. Computer vision models need representations that are more stable than raw pixels.

## Classical Convolution Filters

A convolution filter is a small matrix, often called a kernel, that slides across the image and computes local responses.

Examples:

- blur filters average neighboring pixels;
- sharpen filters emphasize local contrast;
- Sobel filters respond to horizontal or vertical edges.

This is the bridge from classical CV to CNNs: CNNs also use local filters, but learn the filter values from data.

## CNN Intuition

A convolutional neural network learns feature maps.

Early layers often respond to simple visual patterns such as edges, corners, and textures. Deeper layers combine earlier patterns into more task-specific features. Pooling reduces spatial resolution and increases the receptive field, so later units summarize larger parts of the image.

Parameter counting follows the layer formulas:

- `Conv2d`: `out_channels * in_channels * kernel_height * kernel_width + out_channels` when bias is enabled;
- `Linear`: `out_features * in_features + out_features` when bias is enabled.

The practical shows both manual counting and automatic model summaries.

## Detection and Segmentation

Image classification assigns one or more labels to the whole image.

Object detection predicts bounding boxes plus class labels. YOLO-style detectors are one-stage detectors: they predict boxes and classes in one forward pass, which makes them useful for real-time systems.

Semantic segmentation predicts a class for each pixel. It answers a denser question than detection: not only "what object is present?", but also "which pixels belong to that class?"

## Image Embeddings

An image embedding is a vector representation of an image. Instead of comparing raw pixels, we compare vectors produced by a pretrained model.

Useful operations:

- cosine similarity for nearest-neighbor search;
- lower-triangle similarity matrices to avoid duplicate symmetric entries;
- 2D projections such as PCA for rough visualization.

Embeddings are powerful, but they are not perfect semantic truth. They reflect the model architecture, training data, preprocessing, and objective.

## Multimodal Embeddings

Image embeddings become especially interesting when they share a space with text embeddings.

CLIP-style models train image and text encoders together so that matching images and captions are close in the same vector space. This enables zero-shot classification, text-to-image retrieval, image-to-text retrieval, and prompt-driven visual search.

The practical ends by connecting CNN embeddings to this broader multimodal idea.
