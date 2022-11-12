import numpy as np


def l2_normalize(a, axis=-1, order=2):
    #L2 normalization that works for any arbitary axes
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
    
def compute_queryimage_kernel(query_dataset_feat, unlabeled_dataset_feat):
    query_image_sim = []
    unlabeled_feat_norm = l2_normalize(unlabeled_dataset_feat) #l2-normalize the unlabeled feature vector along the feature dimension (batch_size, num_proposals, num_features)
    for i in range(len(query_dataset_feat)):
        query_feat = np.expand_dims(query_dataset_feat[i], axis=0)
        query_feat_norm = l2_normalize(query_feat) #l2-normalize the query feature vector along the feature dimension
        #print(query_feat_norm.shape)
        #print(unlabeled_feat_norm.shape)
        dotp = np.tensordot(query_feat_norm, unlabeled_feat_norm, axes=0) #compute the dot product along the feature dimension, i.e between every GT bbox of rare class in the query image with all proposals from all images in the unlabeled set
        #print(dotp.shape)
        max_match_queryGt_proposal = np.amax(dotp, axis=(1,3)) #find the gt-proposal pair with highest similarity score for each image
        query_image_sim.append(max_match_queryGt_proposal)
    query_image_sim = np.vstack(tuple(query_image_sim))
    print("final query image kernel shape: ", query_image_sim.shape)
    return query_image_sim
