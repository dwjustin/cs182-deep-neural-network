import random

import numpy as np

import torch
import torch.nn.functional as F


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.gard.           #
    ##############################################################################

   
    scores=model(X)
    scores=scores.gather(1,y.view(-1,1)).squeeze()
    
    
    scores.backward(torch.ones(scores.size()))
    
    saliency=X.grad.abs()
    
    saliency, _=torch.max(saliency,dim=1)
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image.
    X_fooling = X.clone().detach().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    
#     output=model.forward(X_fooling)
#     max_output, max_idx=torch.max(output,1)
    
#     while (max_idx!=target_y): 
#         output[:,target_y].backward()
#         grad_img=X_fooling.grad.data
#         norm_grad_img=torch.norm(grad_img,2)
#         dX=learning_rate*(grad_img/norm_grad_img)
#         X_fooling.data+=dX
#         X_fooling.grad=torch.zeros(X_fooling.grad.shape)
            
#         output=model.forward(X_fooling)
#         max_output, max_idx=torch.max(output,1)    
    
#    X_fooling_var=torch.autograd.Variable(X_fooling, requires_grad=True)
    score=model(X_fooling)
    score_idx=score[0,:].argmax(dim=0)
   
    while(score_idx!=target_y):
        score[0,target_y].backward()
        g=X_fooling.grad
        dX=learning_rate*(g/torch.norm(g,2))
        X_fooling.data+=dX
        X_fooling.grad.zero_()
        score=model(X_fooling)
        score_idx=score[0,:].argmax(dim=0)
    
    
#     for i in range(1000):
#         scores=model.forward(X_fooling)
#         _, idx= torch.max(scores,dim=1)
        
#         if(idx!=target_y):
#             scores[:,target_y].backward()
#             dX=learning_rate*(X_fooling.grad.data/torch.norm(X_fooling.grad.data,2))
#             X_fooling.data+=dX.data
#             X_fooling.grad.data.zero_()
#         else:
#             break

#     model.eval()
#     y=torch.LongTensor([target_y])
#     loss=torch.nn.CrossEntropyLoss()
#     for i in range(100):
#         print(i)
#         score=model(X_fooling)
#         max_score=score.argmax(axis=1)
        
#         if max_score==y:
#             break
#         loss_y=loss(score,y)
#         model.zero_grad()
#         loss_y.backward()
#         with torch.no_grad():
#             g=X_fooling.grad
#             dX=learning_rate*g/torch.norm(g)
#             X_fooling-=dX
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling.detach()


def update_class_visulization(model, target_y, l2_reg, learning_rate, img):
    """
    Perform one step of update on a image to maximize the score of target_y
    under a pretrained model.

    Inputs:
    - model: A pretrained CNN that will be used to generate the image
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - img: the image tensor (1, C, H, W) to start from
    """

    # Create a copy of image tensor with gradient support
    img = img.clone().detach().requires_grad_(True)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    scores=model(img)
    score=scores[:,target_y]-(l2_reg*(torch.square(torch.norm(img,p=2))))
    score.backward()
    g=img.grad.data
    img.data+=learning_rate*g
    img.grad.zero_()
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img.detach()
