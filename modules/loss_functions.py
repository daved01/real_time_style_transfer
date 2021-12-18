from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Input


def get_loss_network():
    loss_net = vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=Input(shape=(256,256,3)))
    loss_net_outputs = dict([(layer.name, layer.output) for layer in loss_net.layers])
    loss_net_activations = keras.Model(inputs=loss_net.inputs, outputs=loss_net_outputs)
    return loss_net_activations


def gram_matrix(x):
    """
    Computes the gram matrix with batch dimension.
    
    y = xT * x

    Inputs:
    x  -- tf.tensor with batch dimension (batch_dim, x1, x2, x3)
    """
    x = tf.transpose(x, (0,3,1,2))
    features = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1))
    gram = tf.matmul(features, tf.transpose(features, (0,2,1)))
    return gram


def compute_content_loss(generated, content, dimensions):
    """
    Computes the content loss from the given features.
    Equation 2 in paper.

    Args:
    generated:  Tensor feature map of the generated image.
    content:    Tensor feature map of the content image.
    dimensions: List of layer dimensions [height, width, channels]
    """
    # Check dimensions
    assert generated.shape[0] == content.shape[0], "Batch dimensions of generated and content image don't match!"

    height, width, channels = dimensions[0], dimensions[1], dimensions[2]
    scaling_factor = (int(height/4) * int(width/4) * channels) # H, W, C

    # Sum over all elements, including the batch_size to get average loss over the batch.
    content_reconstruction_loss =  tf.math.reduce_sum(tf.square(generated - content)) / (scaling_factor * generated.shape[0])
    return content_reconstruction_loss


def compute_style_loss(generated, style, dimensions):
    """
    Compute style loss for one layer.
    """
    
    # Dimensions
    height, width, channels = dimensions[0], dimensions[1], dimensions[2] 
    scaling_factor = (channels * height * width)**2
    generated = gram_matrix(generated)
    style = gram_matrix(style)

    # Compute the total average loss over all elements in the batch.
    res = tf.reduce_sum(tf.square(generated - style)) / (scaling_factor * generated.shape[0])
    return res


def compute_perceptual_loss(generated_image, content_image, style_image, loss_net_activations, batch_size, content_layers, style_layers):
    """
    Computes the loss with the loss network.

    Args:
    tf.tensors, scaled to [0,1] with dim (b,h,w,c), RGB.
    """
    
    # Combine input tensors to make one pass with all in parallel.
    input_tensors = tf.concat([generated_image, content_image, style_image], axis=0)

    # Preprocess input_tensors for vgg16. Expects range [0, 255]
    input_tensors = tf.keras.applications.vgg16.preprocess_input(input_tensors*255)

    # Forward pass to get loss from loss network.
    features = loss_net_activations(input_tensors, training=False)

    # Initialize loss
    loss = tf.zeros(shape=())

    # Compute content loss
    for content_layer in content_layers.keys():
        layer_features = features[content_layer]
        generated_features = layer_features[0:batch_size,:,:,:]
        content_features = layer_features[batch_size:2*batch_size,:,:,:]
        loss += compute_content_loss(generated_features, content_features, content_layers[content_layer])

    # Compute style loss
    for style_layer in style_layers.keys():
        layer_features = features[style_layer]
        generated_features = layer_features[0:batch_size,:,:,:]
        style_features = layer_features[2*batch_size,:,:,:]
        style_features = tf.expand_dims(style_features, 0)
        loss += compute_style_loss(generated_features, style_features, style_layers[style_layer])

    return loss


@tf.function
def compute_loss_and_grads(content_image, style_image, transform_network, optimizer, loss_net_activations, batch_size, content_layers, style_layers):
    """
    Takes in content and style images as tf.tensors with batch dimension
    and scaled to range [0,1].
    """
    
    with tf.GradientTape() as tape:

        # Forward pass
        generated_image = transform_network(content_image, training=True)
        
        # Convert to range [0,1]
        generated_image = ((generated_image * 0.5) + 0.5)

        # Get loss
        loss = compute_perceptual_loss(generated_image, content_image, style_image, loss_net_activations, batch_size, content_layers, style_layers)

    # Get gradients and upate weights
    grads = tape.gradient(loss, transform_network.trainable_weights)
    optimizer.apply_gradients(zip(grads, transform_network.trainable_weights))
    return loss