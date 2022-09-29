def getEmbeddings(emb, lake_img_dir, query_img_dir, misclassified_object):
    
    '''query embeddings'''
    query_image_list, query_embeddings = emb.get_embeddings_image_list(query_img_dir, misclassified_object, is_query=False, sel_strategy='avg')

    '''Lake embeddings'''
    lake_image_list, lake_embeddings = emb.get_embeddings_image_list(lake_img_dir, misclassified_object, is_query=False, sel_strategy='avg')
    # print(len(lake_embeddings))
    # print(len(lake_image_list))
    return query_image_list, query_embeddings, lake_image_list, lake_embeddings;